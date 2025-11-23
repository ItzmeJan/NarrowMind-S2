import { stem } from './stem.js';

/**
 * NarrowMind S2 (Statistical 2) Model
 * TF-IDF based sentence ranking system
 */
export class NarrowMindModel {
    constructor(data) {
        this.rawData = data;
        this.tokens = this.parseTokens(data);
        this.sentences = this.parseSentences(data);
        // Use stemmed tokens for corpus documents (for calculations)
        this.corpusDocs = this.sentences.map(s => this.parseTokensStemmed(s));
        // Also store stemmed tokens for IDF calculation
        this.stemmedTokens = this.parseTokensStemmed(data);
        this.idfCache = this.precomputeIDF();
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
     * @returns {string[]} Array of stemmed tokens
     */
    parseTokensStemmed(text) {
        const tokens = this.parseTokens(text);
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
     * @returns {number} Cosine similarity score (0-1)
     */
    calculateTFIDFSimilarity(sentence1, sentence2) {
        // Use stemmed tokens for calculations
        const words1 = this.parseTokensStemmed(sentence1);
        const words2 = this.parseTokensStemmed(sentence2);

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
     * Calculate combined similarity (TF-IDF + Character-level)
     * @param {string} sentence1 - First sentence
     * @param {string} sentence2 - Second sentence
     * @param {number} tfidfWeight - Weight for TF-IDF (default 0.7)
     * @param {number} charWeight - Weight for character similarity (default 0.3)
     * @returns {number} Combined similarity score (0-1)
     */
    calculateCombinedSimilarity(sentence1, sentence2, tfidfWeight = 0.7, charWeight = 0.3) {
        const tfidfScore = this.calculateTFIDFSimilarity(sentence1, sentence2);
        const charScore = this.calculateCharacterSimilarity(sentence1, sentence2);
        
        return (tfidfScore * tfidfWeight) + (charScore * charWeight);
    }

    /**
     * Rank sentences by relevance to a query (using combined TF-IDF + character similarity)
     * @param {string} query - Search query
     * @param {number} topN - Number of top results to return (0 = all)
     * @param {number} tfidfWeight - Weight for TF-IDF similarity (default 0.95)
     * @param {number} charWeight - Weight for character similarity (default 0.05)
     * @returns {Array<[string, number]>} Array of [sentence, score] pairs, sorted by score
     */
    rankSentences(query, topN = 0, tfidfWeight = 0.95, charWeight = 0.05) {
        if (!query || typeof query !== 'string') return [];

        const sentenceRanks = [];
        
        // Use original sentences for output, but stemmed tokens for calculations
        for (const sentence of this.sentences) {
            const similarity = this.calculateCombinedSimilarity(
                query, 
                sentence, 
                tfidfWeight, 
                charWeight
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

