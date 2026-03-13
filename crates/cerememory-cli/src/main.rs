//! Cerememory CLI

use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    println!("Cerememory v{}", env!("CARGO_PKG_VERSION"));
    println!("A living memory database for the age of AI.");
    println!();
    println!("Status: Pre-alpha. Core engine under development.");
    println!("See: https://github.com/co-r-e/cerememory");

    Ok(())
}
