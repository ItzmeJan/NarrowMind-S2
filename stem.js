/**
 * Stemming utility for NarrowMind S2
 * Removes common suffixes from tokens to normalize word forms
 * Improved version with better suffix handling and edge cases
 */

/**
 * Check if a character is a vowel
 * @param {string} char - Character to check
 * @returns {boolean} True if vowel
 */
function isVowel(char) {
    return /[aeiou]/.test(char);
}

/**
 * Check if a character is a consonant
 * @param {string} char - Character to check
 * @returns {boolean} True if consonant
 */
function isConsonant(char) {
    return /[bcdfghjklmnpqrstvwxyz]/.test(char);
}

/**
 * Stem a token by removing common suffixes
 * @param {string} token - Token to stem
 * @returns {string} Stemmed token
 */
export function stem(token) {
    if (!token || token.length < 3) return token?.toLowerCase() || token;
    
    const lowerToken = token.toLowerCase();
    const minStemLength = 2;
    
    // Handle special cases first
    if (lowerToken === 'was' || lowerToken === 'is' || lowerToken === 'are') return lowerToken;
    if (lowerToken === 'has' || lowerToken === 'had' || lowerToken === 'have') return 'hav';
    
    // Multi-step stemming: longer suffixes first, then shorter ones
    // Step 1: Remove longer derivational suffixes
    const longSuffixes = [
        'ational', 'tional', 'ization', 'ation', 'ator', 'alism', 'aliti',
        'ousness', 'iveness', 'fulness', 'ousli', 'ation', 'ator',
        'alism', 'aliti', 'ousness', 'iveness', 'fulness', 'ousli',
        'ement', 'ment', 'able', 'ible', 'ance', 'ence', 'ness',
        'tion', 'sion', 'ing', 'ed', 'er', 'est', 'ly', 'ful', 'less'
    ];
    
    for (const suffix of longSuffixes) {
        if (lowerToken.endsWith(suffix)) {
            const stem = lowerToken.slice(0, -suffix.length);
            if (stem.length >= minStemLength) {
                // Handle e-dropping for -ed, -ing, -er, -est
                if ((suffix === 'ed' || suffix === 'ing' || suffix === 'er' || suffix === 'est') && 
                    stem.length > 1 && stem.endsWith('e')) {
                    return stem.slice(0, -1);
                }
                // Handle consonant doubling (e.g., running -> run)
                if ((suffix === 'ed' || suffix === 'ing') && stem.length > 2) {
                    const lastChar = stem[stem.length - 1];
                    const secondLast = stem[stem.length - 2];
                    if (isConsonant(lastChar) && lastChar === secondLast && 
                        !isVowel(stem[stem.length - 3])) {
                        return stem.slice(0, -1);
                    }
                }
                return stem;
            }
        }
    }
    
    // Step 2: Handle plural forms and simple suffixes
    if (lowerToken.endsWith('ies') && lowerToken.length > 4) {
        return lowerToken.slice(0, -3) + 'y';
    }
    if (lowerToken.endsWith('es') && lowerToken.length > 4) {
        const stem = lowerToken.slice(0, -2);
        // Keep 'e' if preceded by consonant (e.g., 'horses' -> 'horse')
        if (stem.length > 1 && isConsonant(stem[stem.length - 1])) {
            return stem;
        }
        return stem;
    }
    if (lowerToken.endsWith('s') && lowerToken.length > 3 && 
        lowerToken[lowerToken.length - 2] !== 's') { // Don't remove 's' from words ending in 'ss'
        return lowerToken.slice(0, -1);
    }
    
    return lowerToken;
}

