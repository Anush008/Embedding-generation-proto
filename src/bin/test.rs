use regex::Regex;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read the JS file content
    let js_file = "/Users/anush/Desktop/onn/test.ts";
    let content = fs::read_to_string(js_file)?;

    // Use a regular expression to match JavaScript identifiers
    let identifier_regex = r"(?x)
        (?:(?:let|const|var)\s+)          # Optional variable declaration
        ([a-zA-Z_$][a-zA-Z0-9_$]*)         # Identifier
    ";
    let re = Regex::new(identifier_regex)?;

    // Iterate through the matches and store the identifiers
    let mut identifiers: Vec<String> = Vec::new();
    for cap in re.captures_iter(&content) {
        identifiers.push(cap[1].to_string());
    }

    // Print the identifiers
    println!("Identifiers: {:?}", identifiers);

    Ok(())
}
