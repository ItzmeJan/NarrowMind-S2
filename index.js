import fs from "fs";
import { createInterface } from "readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { NarrowMindModel } from "./model.js";

/**
 * Display model configuration and statistics
 * @param {NarrowMindModel} model - The initialized model
 */
function displayModelInfo(model) {
    console.log("\n" + "=".repeat(70));
    console.log("  NarrowMind S2 - Statistical Sentence Ranking Model");
    console.log("=".repeat(70));
    
    // Corpus Statistics
    console.log("\n📊 CORPUS STATISTICS");
    console.log("-".repeat(70));
    console.log(`  Total Sentences:     ${model.sentences.length.toLocaleString()}`);
    console.log(`  Total Tokens:        ${model.tokens.length.toLocaleString()}`);
    console.log(`  Filtered Tokens:     ${model.filteredTokens.length.toLocaleString()} (filler words removed)`);
    console.log(`  Unique Stemmed:      ${new Set(model.stemmedTokens).size.toLocaleString()}`);
    console.log(`  Filler Words Loaded: ${model.fillerWords.size.toLocaleString()}`);
    
    // Co-occurrence Statistics
    const coOccurrencePairs = Array.from(model.coOccurrenceMatrix.values())
        .reduce((sum, map) => sum + map.size, 0);
    console.log(`  Co-occurrence Pairs:  ${coOccurrencePairs.toLocaleString()}`);
    
    // Average sentence length
    const avgSentenceLength = model.sentences.length > 0 
        ? (model.tokens.length / model.sentences.length).toFixed(2)
        : 0;
    console.log(`  Avg Words/Sentence:   ${avgSentenceLength}`);
    
    // Configuration
    console.log("\n⚙️  MODEL CONFIGURATION");
    console.log("-".repeat(70));
    console.log("  Similarity Methods:");
    console.log("    • TF-IDF Cosine Similarity");
    console.log("    • Character-level Similarity (LCS-based)");
    console.log("    • Word Co-occurrence Scoring (Jaccard/PMI)");
    console.log("\n  Features:");
    console.log("    • Stemming (custom suffix-based)");
    console.log("    • Filler word filtering");
    console.log("    • Precomputed IDF cache");
    console.log("    • Co-occurrence matrix");
    
    // Default weights
    console.log("\n  Default Ranking Weights:");
    console.log("    • TF-IDF:           95%");
    console.log("    • Character:        5%");
    console.log("    • Co-occurrence:    0% (disabled by default)");
    
    // Top words by frequency
    const wordFreq = new Map();
    model.stemmedTokens.forEach(token => {
        wordFreq.set(token, (wordFreq.get(token) || 0) + 1);
    });
    const topWords = Array.from(wordFreq.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5)
        .map(([word, count]) => `${word} (${count})`)
        .join(", ");
    
    if (topWords) {
        console.log("\n  Top 5 Most Frequent Words:");
        console.log(`    ${topWords}`);
    }
    
    console.log("\n" + "=".repeat(70) + "\n");
}

/**
 * Main entry point for NarrowMind S2
 */
async function main() {
    // Read input file with error handling
    let data;
    try {
        data = fs.readFileSync("./input.txt", "utf-8");
        if (!data || data.trim().length === 0) {
            console.error("Error: input.txt is empty");
            process.exit(1);
        }
    } catch (error) {
        console.error(`Error reading input.txt: ${error.message}`);
        process.exit(1);
    }

    // Initialize model
    console.log("Initializing NarrowMind S2 model...");
    const model = new NarrowMindModel(data);
    
    // Display model information
    displayModelInfo(model);

    // Create readline interface
    const rl = createInterface({
        input,
        output
    });

    try {
        // Get user query
        const query = await rl.question("=> ");
        
        if (!query || query.trim().length === 0) {
            console.log("No query provided. Exiting.");
            rl.close();
            return;
        }

        // Get token statistics for query tokens
        const queryTokens = model.parseTokens(query.toLowerCase());
        console.log("\n" + "=".repeat(70));
        console.log("  QUERY ANALYSIS");
        console.log("=".repeat(70));
        console.log(`\n  Query: "${query}"`);
        console.log(`  Tokens: ${queryTokens.length}`);
        
        console.log("\n  Token Statistics:");
        console.log("-".repeat(70));
        for (const token of queryTokens) {
            const stats = model.getTokenStats(token);
            const isFiller = model.fillerWords.has(token.toLowerCase());
            const fillerTag = isFiller ? " [FILLER]" : "";
            console.log(`    • ${stats.token}${fillerTag}`);
            console.log(`      Stemmed: ${stats.stemmed} | TF: ${stats.tf.toFixed(4)} | IDF: ${stats.idf.toFixed(4)}`);
            
            // Show top co-occurrences for non-filler words
            if (!isFiller && stats.stemmed) {
                const topCoOcc = model.getTopCoOccurrences(stats.token, 3);
                if (topCoOcc.length > 0) {
                    const coOccStr = topCoOcc.map(([word, count]) => `${word}(${count})`).join(", ");
                    console.log(`      Top Co-occurrences: ${coOccStr}`);
                }
            }
        }

        // Rank sentences
        console.log("\n" + "=".repeat(70));
        console.log("  RANKING RESULTS");
        console.log("=".repeat(70));
        console.log("\n  Configuration:");
        console.log("    • TF-IDF Weight: 95%");
        console.log("    • Character Weight: 5%");
        console.log("    • Co-occurrence: Disabled");
        console.log("    • Filter Fillers: No\n");

        const rankedSentences = model.rankSentences(query);

        // Display results
        if (rankedSentences.length === 0) {
            console.log("  No relevant sentences found.");
        } else {
            console.log(`  Found ${rankedSentences.length} relevant sentence(s):\n`);
            rankedSentences.forEach(([sentence, score], index) => {
                const scoreBar = "█".repeat(Math.floor(score * 20));
                console.log(`  ${index + 1}. [Score: ${score.toFixed(4)}] ${scoreBar}`);
                console.log(`     "${sentence}"\n`);
            });
        }
        
        console.log("=".repeat(70) + "\n");

    } catch (error) {
        console.error(`Error: ${error.message}`);
    } finally {
        rl.close();
    }
}

// Run the application
main().catch(error => {
    console.error(`Fatal error: ${error.message}`);
    process.exit(1);
});
