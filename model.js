import { stem } from './stem.js';
import fs from 'fs';

/**
 * NarrowMind S2 (Statistical 2) Model
 * TF-IDF based sentence ranking system
 */
export class NarrowMindModel {
    constructor(data) {
        this.rawData = data;
        this.tokens = this.parseTokens(data);
        this.sentences = this.parseSentences(data);
        
        // Load filler words
        this.fillerWords = this.loadFillerWords();
        
        // Filter out filler words from tokens
        this.filteredTokens = this.filterTokens(this.tokens);
        
        // Use stemmed tokens for corpus documents (for calculations)
        this.corpusDocs = this.sentences.map(s => this.parseTokensStemmed(s));
        // Also store stemmed tokens for IDF calculation
        this.stemmedTokens = this.parseTokensStemmed(data);
        this.idfCache = this.precomputeIDF();
        
        // Build co-occurrence matrix
        this.coOccurrenceMatrix = this.buildCoOccurrenceMatrix();
    }
    
    /**
     * Load filler words from fillers.json
     * @returns {Set<string>} Set of filler words (lowercase)
     */
    loadFillerWords() {
        try {
            const fillersData = fs.readFileSync('./fillers.json', 'utf-8');
            const fillers = JSON.parse(fillersData);
            // Convert to Set for O(1) lookup and normalize to lowercase
            return new Set(fillers.map(word => word.toLowerCase()));
        } catch (error) {
            console.warn(`Warning: Could not load fillers.json: ${error.message}. Using empty filler list.`);
            return new Set();
        }
    }
    
    /**
     * Filter out filler words from a token array
     * @param {string[]} tokens - Array of tokens to filter
     * @returns {string[]} Filtered array of tokens without filler words
     */
    filterTokens(tokens) {
        if (!tokens || tokens.length === 0) return [];
        return tokens.filter(token => {
            const lowerToken = token.toLowerCase();
            return !this.fillerWords.has(lowerToken);
        });
    }
    
    /**
     * Get filtered tokens from text (parses and filters in one step)
     * @param {string} text - Input text
     * @returns {string[]} Array of filtered tokens (without filler words)
     */
    getFilteredTokens(text) {
        const tokens = this.parseTokens(text);
        return this.filterTokens(tokens);
    }

    /**
     * Parse text into tokens (words) - original tokens for output
     * @param {string} text - Input text
     * @returns {string[]} Array of tokens
     */
    parseTokens(text) {
        if (!text || typeof text !== 'string') return [];
        return text.trim().split(/[^\p{L}\p{N}]+/u).filter(Boolean);
    }

    /**
     * Parse text into stemmed tokens - used for calculations only
     * @param {string} text - Input text
     * @param {boolean} filterFillers - Whether to filter out filler words (default: false)
     * @returns {string[]} Array of stemmed tokens
     */
    parseTokensStemmed(text, filterFillers = false) {
        let tokens = this.parseTokens(text);
        if (filterFillers) {
            tokens = this.filterTokens(tokens);
        }
        return tokens.map(token => stem(token.toLowerCase()));
    }

    /**
     * Parse text into sentences
     * @param {string} text - Input text
     * @returns {string[]} Array of sentences
     */
    parseSentences(text) {
        if (!text || typeof text !== 'string') return [];
        return text.split(/[.!?,"""":;\n]+/)
            .map(s => s.trim())
            .filter(Boolean);
    }

    /**
     * Calculate Term Frequency
     * @param {string} token - Token to calculate TF for
     * @param {string[]} wordList - List of words
     * @returns {number} Term frequency
     */
    calculateTF(token, wordList) {
        if (!wordList || wordList.length === 0) return 0;
        const count = wordList.filter(w => w === token).length;
        return count / wordList.length;
    }

    /**
     * Calculate Inverse Document Frequency
     * @param {string} token - Token to calculate IDF for
     * @param {string[][]} documents - Array of document token arrays
     * @returns {number} Inverse document frequency
     */
    calculateIDF(token, documents) {
        if (!documents || documents.length === 0) return 0;
        const N = documents.length;
        const df = documents.filter(doc => doc.includes(token)).length;
        return Math.log((N + 1) / (df + 1)) + 1;
    }

    /**
     * Precompute IDF values for all tokens in the corpus (using stemmed tokens)
     * @returns {Map<string, number>} Map of token to IDF value
     */
    precomputeIDF() {
        const idfMap = new Map();
        const allTokens = [...new Set(this.stemmedTokens)];
        
        for (const token of allTokens) {
            idfMap.set(token, this.calculateIDF(token, this.corpusDocs));
        }
        
        return idfMap;
    }

    /**
     * Get IDF value for a token (uses cache if available)
     * @param {string} token - Token to get IDF for
     * @returns {number} IDF value
     */
    getIDF(token) {
        if (this.idfCache.has(token)) {
            return this.idfCache.get(token);
        }
        // If token not in cache, calculate and cache it
        const idf = this.calculateIDF(token, this.corpusDocs);
        this.idfCache.set(token, idf);
        return idf;
    }

    /**
     * Calculate character-level similarity using Levenshtein distance
     * @param {string} str1 - First string
     * @param {string} str2 - Second string
     * @returns {number} Similarity score (0-1), where 1 is identical
     */
    calculateCharacterSimilarity(str1, str2) {
        if (!str1 || !str2) return 0;
        if (str1 === str2) return 1;
        
        const s1 = str1.toLowerCase();
        const s2 = str2.toLowerCase();
        
        // Use longest common subsequence (LCS) for better character-level similarity
        const lcsLength = this.longestCommonSubsequence(s1, s2);
        const maxLength = Math.max(s1.length, s2.length);
        
        if (maxLength === 0) return 1;
        return lcsLength / maxLength;
    }

    /**
     * Calculate longest common subsequence length
     * @param {string} str1 - First string
     * @param {string} str2 - Second string
     * @returns {number} Length of LCS
     */
    longestCommonSubsequence(str1, str2) {
        const m = str1.length;
        const n = str2.length;
        const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
        
        for (let i = 1; i <= m; i++) {
            for (let j = 1; j <= n; j++) {
                if (str1[i - 1] === str2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }
        
        return dp[m][n];
    }

    /**
     * Calculate TF-IDF cosine similarity between two sentences (using stemmed tokens)
     * @param {string} sentence1 - First sentence
     * @param {string} sentence2 - Second sentence
     * @param {boolean} filterFillers - Whether to filter out filler words (default: false)
     * @returns {number} Cosine similarity score (0-1)
     */
    calculateTFIDFSimilarity(sentence1, sentence2, filterFillers = false) {
        // Use stemmed tokens for calculations
        const words1 = this.parseTokensStemmed(sentence1, filterFillers);
        const words2 = this.parseTokensStemmed(sentence2, filterFillers);

        if (words1.length === 0 || words2.length === 0) return 0;

        const vocab = [...new Set([...words1, ...words2])];

        const vec1 = vocab.map(token => 
            this.calculateTF(token, words1) * this.getIDF(token)
        );
        const vec2 = vocab.map(token => 
            this.calculateTF(token, words2) * this.getIDF(token)
        );

        // Calculate cosine similarity
        const dot = vec1.reduce((sum, val, i) => sum + val * vec2[i], 0);
        const mag1 = Math.sqrt(vec1.reduce((sum, val) => sum + val * val, 0));
        const mag2 = Math.sqrt(vec2.reduce((sum, val) => sum + val * val, 0));

        if (mag1 === 0 || mag2 === 0) return 0;
        return dot / (mag1 * mag2);
    }

    /**
     * Calculate combined similarity (TF-IDF + Character-level + optional Co-occurrence)
     * @param {string} sentence1 - First sentence
     * @param {string} sentence2 - Second sentence
     * @param {number} tfidfWeight - Weight for TF-IDF (default 0.7)
     * @param {number} charWeight - Weight for character similarity (default 0.3)
     * @param {boolean} filterFillers - Whether to filter out filler words (default: false)
     * @param {number} coOccurrenceWeight - Weight for co-occurrence similarity (default: 0, disabled)
     * @param {string} coOccurrenceMethod - Co-occurrence method: 'jaccard' or 'pmi' (default: 'jaccard')
     * @returns {number} Combined similarity score (0-1)
     */
    calculateCombinedSimilarity(sentence1, sentence2, tfidfWeight = 0.7, charWeight = 0.3, filterFillers = false, coOccurrenceWeight = 0, coOccurrenceMethod = 'jaccard') {
        const tfidfScore = this.calculateTFIDFSimilarity(sentence1, sentence2, filterFillers);
        const charScore = this.calculateCharacterSimilarity(sentence1, sentence2);
        
        let totalWeight = tfidfWeight + charWeight;
        let score = (tfidfScore * tfidfWeight) + (charScore * charWeight);
        
        // Add co-occurrence score if enabled
        if (coOccurrenceWeight > 0) {
            const coOccurrenceScore = this.calculateCoOccurrenceSimilarity(sentence1, sentence2, filterFillers, coOccurrenceMethod);
            score += coOccurrenceScore * coOccurrenceWeight;
            totalWeight += coOccurrenceWeight;
        }
        
        // Normalize by total weight
        return totalWeight > 0 ? score / totalWeight : 0;
    }

    /**
     * Rank sentences by relevance to a query (using combined TF-IDF + character similarity + optional co-occurrence)
     * @param {string} query - Search query
     * @param {number} topN - Number of top results to return (0 = all)
     * @param {number} tfidfWeight - Weight for TF-IDF similarity (default 0.95)
     * @param {number} charWeight - Weight for character similarity (default 0.05)
     * @param {boolean} filterWords - Whether to filter out filler words (default: false)
     * @param {number} coOccurrenceWeight - Weight for co-occurrence similarity (default: 0, disabled)
     * @param {string} coOccurrenceMethod - Co-occurrence method: 'jaccard' or 'pmi' (default: 'jaccard')
     * @returns {Array<[string, number]>} Array of [sentence, score] pairs, sorted by score
     */
    rankSentences(query, topN = 0, tfidfWeight = 0.95, charWeight = 0.05, filterWords = false, coOccurrenceWeight = 0, coOccurrenceMethod = 'jaccard') {
        if (!query || typeof query !== 'string') return [];

        const sentenceRanks = [];
        
        // Use original sentences for output, but stemmed tokens for calculations
        for (const sentence of this.sentences) {
            const similarity = this.calculateCombinedSimilarity(
                query, 
                sentence, 
                tfidfWeight, 
                charWeight,
                filterWords,
                coOccurrenceWeight,
                coOccurrenceMethod
            );
            if (similarity > 0) {
                sentenceRanks.push([sentence, similarity]);
            }
        }

        // Sort by score (descending)
        sentenceRanks.sort((a, b) => b[1] - a[1]);

        // Return top N if specified
        return topN > 0 ? sentenceRanks.slice(0, topN) : sentenceRanks;
    }

    /**
     * Get TF value for a token in the corpus (using stemmed tokens)
     * @param {string} token - Token to get TF for
     * @returns {number} Term frequency
     */
    getTF(token) {
        const stemmedToken = stem(token.toLowerCase());
        return this.calculateTF(stemmedToken, this.stemmedTokens);
    }

    /**
     * Get statistics for a query token
     * @param {string} token - Token to analyze
     * @returns {Object} Object with TF and IDF values (using stemmed version)
     */
    getTokenStats(token) {
        const normalizedToken = token.toLowerCase();
        const stemmedToken = stem(normalizedToken);
        return {
            token: normalizedToken,
            stemmed: stemmedToken,
            tf: this.getTF(normalizedToken),
            idf: this.getIDF(stemmedToken)
        };
    }

    /**
     * Build co-occurrence matrix for words appearing in the same sentences
     * @returns {Map<string, Map<string, number>>} Nested map: word1 -> word2 -> count
     */
    buildCoOccurrenceMatrix() {
        const matrix = new Map();
        
        // Process each sentence
        for (const sentenceTokens of this.corpusDocs) {
            const uniqueTokens = [...new Set(sentenceTokens)];
            
            // Count co-occurrences within the sentence
            for (let i = 0; i < uniqueTokens.length; i++) {
                const word1 = uniqueTokens[i];
                
                if (!matrix.has(word1)) {
                    matrix.set(word1, new Map());
                }
                
                // Count co-occurrence with all other words in the sentence
                for (let j = 0; j < uniqueTokens.length; j++) {
                    if (i !== j) {
                        const word2 = uniqueTokens[j];
                        const currentCount = matrix.get(word1).get(word2) || 0;
                        matrix.get(word1).set(word2, currentCount + 1);
                    }
                }
            }
        }
        
        return matrix;
    }
    
    /**
     * Get co-occurrence count between two words
     * @param {string} word1 - First word (stemmed)
     * @param {string} word2 - Second word (stemmed)
     * @returns {number} Co-occurrence count
     */
    getCoOccurrenceCount(word1, word2) {
        const stemmed1 = stem(word1.toLowerCase());
        const stemmed2 = stem(word2.toLowerCase());
        
        if (!this.coOccurrenceMatrix.has(stemmed1)) {
            return 0;
        }
        
        return this.coOccurrenceMatrix.get(stemmed1).get(stemmed2) || 0;
    }
    
    /**
     * Calculate co-occurrence score between two words (normalized)
     * Uses Jaccard similarity or pointwise mutual information
     * @param {string} word1 - First word
     * @param {string} word2 - Second word
     * @param {string} method - Scoring method: 'jaccard' or 'pmi' (default: 'jaccard')
     * @returns {number} Co-occurrence score (0-1 for jaccard, unbounded for PMI)
     */
    calculateCoOccurrenceScore(word1, word2, method = 'jaccard') {
        const stemmed1 = stem(word1.toLowerCase());
        const stemmed2 = stem(word2.toLowerCase());
        
        if (stemmed1 === stemmed2) return 1;
        
        const coOccurrence = this.getCoOccurrenceCount(stemmed1, stemmed2);
        
        if (coOccurrence === 0) return 0;
        
        if (method === 'jaccard') {
            // Jaccard similarity: intersection / union
            const word1Sentences = new Set();
            const word2Sentences = new Set();
            
            // Find sentences containing each word
            for (let i = 0; i < this.corpusDocs.length; i++) {
                if (this.corpusDocs[i].includes(stemmed1)) {
                    word1Sentences.add(i);
                }
                if (this.corpusDocs[i].includes(stemmed2)) {
                    word2Sentences.add(i);
                }
            }
            
            const intersection = new Set([...word1Sentences].filter(x => word2Sentences.has(x)));
            const union = new Set([...word1Sentences, ...word2Sentences]);
            
            return union.size > 0 ? intersection.size / union.size : 0;
            
        } else if (method === 'pmi') {
            // Pointwise Mutual Information: log(P(x,y) / (P(x) * P(y)))
            const totalSentences = this.corpusDocs.length;
            const word1Count = this.corpusDocs.filter(doc => doc.includes(stemmed1)).length;
            const word2Count = this.corpusDocs.filter(doc => doc.includes(stemmed2)).length;
            const coOccurrenceCount = this.getCoOccurrenceCount(stemmed1, stemmed2);
            
            if (word1Count === 0 || word2Count === 0 || coOccurrenceCount === 0) return 0;
            
            const pxy = coOccurrenceCount / totalSentences;
            const px = word1Count / totalSentences;
            const py = word2Count / totalSentences;
            
            return Math.log2((pxy + 0.0001) / (px * py + 0.0001)); // Add small epsilon to avoid log(0)
        }
        
        return 0;
    }
    
    /**
     * Calculate word co-occurrence similarity between two sentences
     * @param {string} sentence1 - First sentence
     * @param {string} sentence2 - Second sentence
     * @param {boolean} filterFillers - Whether to filter out filler words
     * @param {string} method - Co-occurrence scoring method: 'jaccard' or 'pmi'
     * @returns {number} Co-occurrence similarity score (0-1)
     */
    calculateCoOccurrenceSimilarity(sentence1, sentence2, filterFillers = false, method = 'jaccard') {
        const words1 = this.parseTokensStemmed(sentence1, filterFillers);
        const words2 = this.parseTokensStemmed(sentence2, filterFillers);
        
        if (words1.length === 0 || words2.length === 0) return 0;
        
        const uniqueWords1 = [...new Set(words1)];
        const uniqueWords2 = [...new Set(words2)];
        
        let totalScore = 0;
        let pairCount = 0;
        
        // Calculate average co-occurrence score between all word pairs
        for (const word1 of uniqueWords1) {
            for (const word2 of uniqueWords2) {
                const score = this.calculateCoOccurrenceScore(word1, word2, method);
                totalScore += score;
                pairCount++;
            }
        }
        
        // Normalize to 0-1 range (for PMI, we'll use a sigmoid-like normalization)
        if (method === 'pmi') {
            // Normalize PMI scores (typically range from -inf to +inf, but usually -5 to +5)
            return Math.max(0, Math.min(1, (totalScore / pairCount + 5) / 10));
        }
        
        return pairCount > 0 ? totalScore / pairCount : 0;
    }
    
    /**
     * Get top co-occurring words for a given word
     * @param {string} word - Word to find co-occurrences for
     * @param {number} topN - Number of top co-occurring words to return
     * @returns {Array<[string, number]>} Array of [word, count] pairs, sorted by count
     */
    getTopCoOccurrences(word, topN = 10) {
        const stemmedWord = stem(word.toLowerCase());
        
        if (!this.coOccurrenceMatrix.has(stemmedWord)) {
            return [];
        }
        
        const coOccurrences = Array.from(this.coOccurrenceMatrix.get(stemmedWord).entries())
            .map(([word, count]) => [word, count])
            .sort((a, b) => b[1] - a[1])
            .slice(0, topN);
        
        return coOccurrences;
    }

    ngram(n=2) {
        const map = [];
        for (var i = 0; i < this.tokens.length-n; i++) {
            const current = []
            for (var j = 1; j <= n; j++) {
                const cw = this.stemmedTokens[i+j];
                current.push(cw);
            }
            map.push(current);
        }
        return map;
    }
}

