/**
 * Stemming utility for NarrowMind S2
 * Removes common suffixes from tokens to normalize word forms
 */

/**
 * Stem a token by removing common suffixes
 * @param {string} token - Token to stem
 * @returns {string} Stemmed token
 */
export function stem(token) {
    if (!token || token.length < 3) return token;
    
    const lowerToken = token.toLowerCase();
    
    // Remove common suffixes (order matters - longer suffixes first)
    const suffixes = [
        'ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 
        'ness', 'ment', 'able', 'ible', 'ful', 'less',
        's', 'es', 'ies'
    ];
    
    for (const suffix of suffixes) {
        if (lowerToken.endsWith(suffix) && lowerToken.length > suffix.length + 2) {
            return lowerToken.slice(0, -suffix.length);
        }
    }
    
    return lowerToken;
}

