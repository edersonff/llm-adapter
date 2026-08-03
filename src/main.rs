use std::io::Write;

use clap::Parser;
use futures_util::StreamExt;
use llm_adapter::{Client, LlmAdapterError, Message};

#[derive(Parser)]
#[command(name = "llm-adapter", about = "LLM adapter CLI")]
struct Cli {
    #[arg(short, long, default_value = "config.yaml")]
    config: String,

    #[arg(short, long)]
    model: Option<String>,

    #[arg(short, long)]
    stream: bool,

    #[arg(short, long)]
    verbose: bool,
}

fn init_tracing(verbose: bool) {
    let level = if verbose { "info" } else { "error" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level)),
        )
        .with_writer(std::io::stderr)
        .with_target(false)
        .with_timer(tracing_subscriber::fmt::time::time())
        .init();
}

fn exit_code(err: &LlmAdapterError) -> i32 {
    match err {
        LlmAdapterError::ConfigError { .. }
        | LlmAdapterError::ModelNotFound { .. }
        | LlmAdapterError::RequestValidation { .. } => 1,
        LlmAdapterError::ApiError { .. }
        | LlmAdapterError::FallbackExhausted { .. }
        | LlmAdapterError::AllKeysExhausted { .. }
        | LlmAdapterError::TimeoutError { .. }
        | LlmAdapterError::HttpError { .. }
        | LlmAdapterError::StreamError { .. } => 2,
    }
}

fn main() {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    let input = match std::io::read_to_string(std::io::stdin()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("failed to read stdin: {e}");
            std::process::exit(1);
        }
    };

    let json: serde_json::Value = match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("failed to parse stdin JSON: {e}");
            std::process::exit(1);
        }
    };

    let model = cli
        .model
        .or_else(|| json["model"].as_str().map(String::from))
        .unwrap_or_default();

    let messages: Vec<Message> = match serde_json::from_value(json["messages"].clone()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to parse messages: {e}");
            std::process::exit(1);
        }
    };

    let temperature = json["temperature"].as_f64();
    let max_tokens = json["max_tokens"].as_u64().map(|v| v as u32);

    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to create runtime: {e}");
            std::process::exit(1);
        }
    };

    let config_path = cli.config.clone();
    let result = rt.block_on(async {
        let client = Client::from_config_file(&config_path).await?;
        let mut builder = client.chat().model(&model).messages(messages);
        if let Some(t) = temperature {
            builder = builder.temperature(t);
        }
        if let Some(m) = max_tokens {
            builder = builder.max_tokens(m);
        }

        if cli.stream {
            let mut stream = builder.send_stream().await?;
            let mut stdout = std::io::stdout().lock();
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let s = serde_json::to_string(&chunk)
                            .map_err(|e| LlmAdapterError::StreamError {
                                provider: String::new(),
                                message: e.to_string(),
                            })?;
                        writeln!(stdout, "data: {s}").map_err(|e| {
                            LlmAdapterError::StreamError {
                                provider: String::new(),
                                message: e.to_string(),
                            }
                        })?;
                        stdout.flush().map_err(|e| LlmAdapterError::StreamError {
                            provider: String::new(),
                            message: e.to_string(),
                        })?;
                    }
                    Err(e) => {
                        eprintln!("stream error: {e}");
                        break;
                    }
                }
            }
            writeln!(stdout, "data: [DONE]").map_err(|e| LlmAdapterError::StreamError {
                provider: String::new(),
                message: e.to_string(),
            })?;
            stdout.flush().map_err(|e| LlmAdapterError::StreamError {
                provider: String::new(),
                message: e.to_string(),
            })?;
            Ok(())
        } else {
            let response = builder.send().await?;
            let output = serde_json::to_string(&response).map_err(|e| {
                LlmAdapterError::StreamError {
                    provider: String::new(),
                    message: e.to_string(),
                }
            })?;
            println!("{output}");
            Ok(())
        }
    });

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(exit_code(&e));
    }
}
