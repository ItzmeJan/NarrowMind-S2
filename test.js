import fs from "fs";
import { createInterface } from "readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { NarrowMindModel } from "./model.js";

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
    const model = new NarrowMindModel(data);

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
        console.log("\nToken Statistics:");
        console.log("-".repeat(50));
        for (const token of queryTokens) {
            const stats = model.getTokenStats(token);
            console.log(`${stats.token} (stemmed: ${stats.stemmed}): TF=${stats.tf.toFixed(4)}, IDF=${stats.idf.toFixed(4)}`);
        }

        // Rank sentences
        console.log("\n" + "=".repeat(50));
        console.log("Ranking sentences...");
        console.log("=".repeat(50) + "\n");

        const rankedSentences = model.rankSentences(query);

        // Display results
        if (rankedSentences.length === 0) {
            console.log("No relevant sentences found.");
        } else {
            console.log(`Found ${rankedSentences.length} relevant sentence(s):\n`);
            rankedSentences.forEach(([sentence, score], index) => {
                console.log(`${index + 1}. [Score: ${score.toFixed(4)}]`);
                console.log(`   ${sentence}\n`);
            });
        }
        const map = model.ngram(3);
        console.log(map);
        const filtered = map.filter(a => a.includes('she'));
        console.log(filtered);
    } catch (error) {
        console.error(`Error: ${error.message}`);
    } finally {
        rl.close();
    }
}

main().catch(error => {
    console.error(`Fatal error: ${error.message}`);
    process.exit(1);
});
