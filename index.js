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
    console.log("  NarrowMind S2 - Statistical Language Model");
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
    console.log("  Language Model Components:");
    console.log("    • Statistical text analysis and understanding");
    console.log("    • Multi-metric similarity scoring");
    console.log("    • Semantic relationship modeling");
    console.log("\n  Similarity Metrics (used in scoring):");
    console.log("    • TF-IDF Cosine Similarity");
    console.log("    • Character-level Similarity (LCS-based)");
    console.log("    • Word Co-occurrence Scoring (Jaccard/PMI)");
    console.log("\n  Features:");
    console.log("    • Stemming (custom suffix-based)");
    console.log("    • Filler word filtering");
    console.log("    • Precomputed IDF cache");
    console.log("    • Co-occurrence matrix");
    console.log("    • N-gram analysis");
    
    // Default weights
    console.log("\n  Scoring Weight Configuration:");
    console.log("    • TF-IDF:           70%");
    console.log("    • Character:       10%");
    console.log("    • Co-occurrence:   20% (Jaccard)");
    console.log("    Note: Sentence ranking uses weighted combination of these metrics");
    
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
 * Main entry point for NarrowMind S2 Language Model
 * Interactive shell for querying and analyzing text using the language model
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

    // Initialize language model
    console.log("Initializing NarrowMind S2 Language Model...");
    const model = new NarrowMindModel(data);
    
    // Display model information
    displayModelInfo(model);

    // Create readline interface
    const rl = createInterface({
        input,
        output
    });

    // Process a single query
    async function processQuery(query) {
        if (!query || query.trim().length === 0) {
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
        
        // Collect co-occurrences for finding common ones
        const allCoOccurrences = new Map(); // word -> Set of query tokens it co-occurs with
        
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
                    
                    // Track co-occurrences for common analysis
                    topCoOcc.forEach(([word, count]) => {
                        if (!allCoOccurrences.has(word)) {
                            allCoOccurrences.set(word, { tokens: new Set(), totalCount: 0 });
                        }
                        allCoOccurrences.get(word).tokens.add(stats.token);
                        allCoOccurrences.get(word).totalCount += count;
                    });
                }
            }
        }
        
        // Find and display common co-occurrences
        const commonCoOccurrences = Array.from(allCoOccurrences.entries())
            .filter(([word, data]) => data.tokens.size > 1) // Appears with multiple query tokens
            .map(([word, data]) => ({
                word,
                tokenCount: data.tokens.size,
                totalCount: data.totalCount,
                tokens: Array.from(data.tokens)
            }))
            .sort((a, b) => {
                // Sort by number of tokens first, then by total count
                if (b.tokenCount !== a.tokenCount) {
                    return b.tokenCount - a.tokenCount;
                }
                return b.totalCount - a.totalCount;
            });
        
        if (commonCoOccurrences.length > 0) {
            console.log("\n  Common Co-occurrences (across multiple query tokens):");
            console.log("-".repeat(70));
            commonCoOccurrences.forEach(({ word, tokenCount, totalCount, tokens }, index) => {
                const tokensStr = tokens.join(", ");
                console.log(`    ${index + 1}. ${word} - co-occurs with ${tokenCount} query token${tokenCount > 1 ? 's' : ''} (${tokensStr}) [total: ${totalCount}]`);
            });
        }

        // Find most common token from query n-grams
        console.log("\n  N-gram Analysis:");
        console.log("-".repeat(70));
        const mostCommonTokens = model.findMostCommonTokenFromQueryNgrams(query, 2, false, 5);
        if (mostCommonTokens.length > 0) {
            console.log("  Most Common Tokens from Query N-grams (Bigrams):");
            mostCommonTokens.forEach(([token, count], index) => {
                console.log(`    ${index + 1}. ${token} (appears in ${count} matching n-gram${count !== 1 ? 's' : ''})`);
            });
        } else {
            console.log("  No matching n-grams found in corpus.");
        }

        // Rank sentences using language model scoring
        console.log("\n" + "=".repeat(70));
        console.log("  LANGUAGE MODEL SCORING RESULTS");
        console.log("=".repeat(70));
        console.log("\n  Scoring Configuration:");
        console.log("    • TF-IDF Weight: 70%");
        console.log("    • Character Weight: 10%");
        console.log("    • Co-occurrence: 20% (Jaccard)");
        console.log("    • Filter Fillers: No");
        console.log("    Note: Results ranked using weighted combination of similarity metrics\n");

        const rankedSentences = model.rankSentences(query, 0, 0.70, 0.10, false, 0.20, 'jaccard');

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
    }

    // Recursive shell loop
    try {
        console.log("\nType 'exit', 'quit', or 'q' to exit the shell.\n");
        
        while (true) {
            const query = await rl.question("=> ");
            
            // Check for exit commands
            const trimmedQuery = query.trim().toLowerCase();
            if (trimmedQuery === 'exit' || trimmedQuery === 'quit' || trimmedQuery === 'q') {
                console.log("\nExiting NarrowMind S2 Language Model. Goodbye!\n");
                break;
            }
            
            // Process the query
            try {
                await processQuery(query);
            } catch (error) {
                console.error(`\nError processing query: ${error.message}\n`);
            }
        }
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
