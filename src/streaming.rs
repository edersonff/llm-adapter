use std::pin::Pin;

use async_stream::try_stream;
use futures_core::Stream;
use tokio::io::AsyncBufReadExt;

use crate::models::StreamChunk;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SseEvent {
    Message { data: String },
    Done,
    KeepAlive,
}

#[derive(Debug, PartialEq, thiserror::Error)]
pub enum SseError {
    #[error("SSE parse error: {message}")]
    Parse { message: String },
    #[error("JSON deserialize error: {message} (raw: {raw})")]
    Json { message: String, raw: String },
}

pub struct SseParser {
    buffer: String,
    event_type: Option<String>,
    data_lines: Vec<String>,
    bom_stripped: bool,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            event_type: None,
            data_lines: Vec::new(),
            bom_stripped: false,
        }
    }

    fn strip_bom(&mut self) {
        if self.bom_stripped {
            return;
        }
        let bom: &[u8] = &[0xEF, 0xBB, 0xBF];
        if self.buffer.starts_with("\u{FEFF}")
            || (self.buffer.len() >= 3 && self.buffer.as_bytes()[..3] == *bom)
        {
            self.buffer = self.buffer[3..].to_string();
        }
        self.bom_stripped = true;
    }

    pub fn feed(&mut self, chunk: &str) -> Vec<Result<SseEvent, SseError>> {
        self.buffer.push_str(chunk);
        self.strip_bom();

        let mut events = Vec::new();

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos]
                .trim_end_matches('\r')
                .to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if line.is_empty() {
                if !self.data_lines.is_empty() {
                    let data = self.data_lines.join("\n");
                    self.data_lines.clear();
                    self.event_type = None;

                    if data == "[DONE]" {
                        events.push(Ok(SseEvent::Done));
                    } else {
                        events.push(Ok(SseEvent::Message { data }));
                    }
                }
            } else if line.starts_with(':') {
                events.push(Ok(SseEvent::KeepAlive));
            } else if let Some(data) = line.strip_prefix("data:") {
                let trimmed = data.trim_start().to_string();
                if !trimmed.is_empty() {
                    self.data_lines.push(trimmed);
                }
            } else if let Some(event_type) = line.strip_prefix("event:") {
                self.event_type = Some(event_type.trim().to_string());
            }
        }

        events
    }

    pub fn parse_message_to_chunk(data: &str) -> Result<StreamChunk, SseError> {
        serde_json::from_str(data).map_err(|e| SseError::Json {
            message: e.to_string(),
            raw: data.to_string(),
        })
    }
}

pub fn parse_sse_stream(
    mut body: impl tokio::io::AsyncBufRead + Unpin + Send + 'static,
) -> Pin<Box<dyn Stream<Item = Result<StreamChunk, SseError>> + Send>> {
    let s = try_stream! {
        let mut parser = SseParser::new();
        let mut buf = String::new();

        loop {
            buf.clear();
            let n = body.read_line(&mut buf).await.map_err(|e| SseError::Parse {
                message: e.to_string(),
            })?;
            if n == 0 {
                break;
            }
            let events = parser.feed(&buf);
            for event in events {
                match event {
                    Ok(SseEvent::Message { data }) => {
                        let chunk = SseParser::parse_message_to_chunk(&data)?;
                        yield chunk;
                    }
                    Ok(SseEvent::Done) => {
                        return;
                    }
                    Ok(SseEvent::KeepAlive) => {}
                    Err(e) => Err(e)?,
                }
            }
        }

        let remaining = parser.feed("");
        for event in remaining {
            match event {
                Ok(SseEvent::Message { data }) => {
                    let chunk = SseParser::parse_message_to_chunk(&data)?;
                    yield chunk;
                }
                Ok(SseEvent::Done) => {
                    return;
                }
                Ok(SseEvent::KeepAlive) => {}
                Err(e) => Err(e)?,
            }
        }
    };

    Box::pin(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_message_event() {
        let mut parser = SseParser::new();
        let input = r#"data: {"id":"1","choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}"#;
        let events = parser.feed(&format!("{}\n\n", input));
        assert_eq!(events.len(), 1);
        match &events[0] {
            Ok(SseEvent::Message { data }) => {
                let chunk: StreamChunk = serde_json::from_str(data).unwrap();
                assert_eq!(chunk.id.as_deref(), Some("1"));
                assert_eq!(chunk.choices[0].delta.content.as_deref(), Some("hello"));
            }
            other => panic!("expected Message, got {:?}", other),
        }
    }

    #[test]
    fn done_sentinel() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: [DONE]\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], Ok(SseEvent::Done));
    }

    #[test]
    fn split_frame() {
        let mut parser = SseParser::new();
        let e1 = parser.feed("data: {\"id\":");
        assert!(e1.is_empty());
        let e2 = parser.feed("\"2\"}\n\n");
        assert_eq!(e2.len(), 1);
        match &e2[0] {
            Ok(SseEvent::Message { data }) => {
                let chunk: StreamChunk = serde_json::from_str(data).unwrap();
                assert_eq!(chunk.id.as_deref(), Some("2"));
            }
            other => panic!("expected Message, got {:?}", other),
        }
    }

    #[test]
    fn comment_line() {
        let mut parser = SseParser::new();
        let events = parser.feed(": this is a comment\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0], Ok(SseEvent::KeepAlive));
    }

    #[test]
    fn multiline_data() {
        let mut parser = SseParser::new();
        let events = parser.feed("data: line1\ndata: line2\n\n");
        assert_eq!(events.len(), 1);
        match &events[0] {
            Ok(SseEvent::Message { data }) => {
                assert_eq!(data, "line1\nline2");
            }
            other => panic!("expected Message, got {:?}", other),
        }
    }

    #[test]
    fn malformed_json_returns_error() {
        let result = SseParser::parse_message_to_chunk("{broken}");
        assert!(result.is_err());
        match result.unwrap_err() {
            SseError::Json { message, raw } => {
                assert!(!message.is_empty());
                assert_eq!(raw, "{broken}");
            }
            other => panic!("expected Json error, got {:?}", other),
        }
    }

    #[test]
    fn crlf_line_endings() {
        let mut parser = SseParser::new();
        let input = "data: {\"id\":\"3\"}\r\n\r\n";
        let events = parser.feed(input);
        assert_eq!(events.len(), 1);
        match &events[0] {
            Ok(SseEvent::Message { data }) => {
                let chunk: StreamChunk = serde_json::from_str(data).unwrap();
                assert_eq!(chunk.id.as_deref(), Some("3"));
            }
            other => panic!("expected Message, got {:?}", other),
        }
    }

    #[test]
    fn empty_data_line_skipped() {
        let mut parser = SseParser::new();
        let events = parser.feed("data:\n\n");
        assert!(events.is_empty());
    }

    #[test]
    fn unknown_fields_ignored() {
        let mut parser = SseParser::new();
        let events = parser.feed("id: 123\nevent: message\nretry: 5000\n\n");
        assert!(events.is_empty());
    }

    #[test]
    fn multiple_events_in_single_feed() {
        let mut parser = SseParser::new();
        let input = "data: first\n\ndata: second\n\n";
        let events = parser.feed(input);
        assert_eq!(events.len(), 2);
        assert_eq!(
            events[0],
            Ok(SseEvent::Message {
                data: "first".to_string()
            })
        );
        assert_eq!(
            events[1],
            Ok(SseEvent::Message {
                data: "second".to_string()
            })
        );
    }
}
