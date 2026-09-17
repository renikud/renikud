/*
wget https://huggingface.co/renikud/renikud/resolve/main/model.onnx
cargo run --example basic
*/

use renikud_rs::G2P;

fn main() -> anyhow::Result<()> {
    let mut g2p = G2P::new("model.onnx")?;
    println!("{}", g2p.phonemize("את רוצה לבוא? אתה רוצה לבוא? אתם רוצים? מה דעתכם?")?);
    Ok(())
}
