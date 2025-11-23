import fs from "fs";
import {createInterface} from "readline/promises";
import { stdin as input, stdout as output } from "node:process";

const intf = createInterface({
    input,
    output
});

const data = fs.readFileSync("./input.txt").toString();

function parseTokens(data) {
    const tokens = data.trim().split(/[^\p{L}\p{N}]+/u).filter(Boolean);
    return tokens;
}

function parseSentences(data) {
    const s = data.split(/[.!?,"”“:;\n]+/).map(s => s.trim()).filter(Boolean);
    return s;
}

function TFIDF(s1, s2,corpusDocs) {
    const words1 = parseTokens(s1.toLowerCase());
    const words2 = parseTokens(s2.toLowerCase());

    const vocab = [...new Set([...words1, ...words2])];

    const vec1 = vocab.map(t => TF(t, words1) * IDF(t, corpusDocs));
    const vec2 = vocab.map(t => TF(t, words2) * IDF(t, corpusDocs));

    // cosine similarity
    const dot = vec1.reduce((s, v, i) => s + v * vec2[i], 0);
    const mag1 = Math.sqrt(vec1.reduce((s, v) => s + v*v, 0));
    const mag2 = Math.sqrt(vec2.reduce((s, v) => s + v*v, 0));

    if (mag1 === 0 || mag2 === 0) return 0;
    return dot / (mag1 * mag2);
}


function TF(token,wordlist) {
    let count = wordlist.filter(w => w==token).length;
    return count/wordlist.length;
}

function IDF(token, documents) {
    const N = documents.length;
    const df = documents.filter(doc => doc.includes(token)).length;
    return Math.log((N + 1) / (df + 1)) + 1;
}

function ngrams() {

}
const tokens = parseTokens(data);
const sentences = parseSentences(data);
const corpusDocs = sentences.map(parseTokens);  // array of docs

console.log(tokens);
console.log(sentences)
console.log("TF(she):",TF('the',tokens));
console.log("IDF(the):", IDF("the", corpusDocs));


const words = await intf.question("=> ");
const qtokens = parseTokens(words.toLowerCase());
for (const token of qtokens) {
    console.log(token,TF(token,tokens));
    console.log(token,IDF(token,sentences.map(parseTokens)));
}
let sentenceRanks = [];
for(const sentence of sentences) {
    console.log()
    console.log('words:',words);
    console.log('Sentence: ',sentence);
    const tfidf = TFIDF(words, sentence, corpusDocs);
    console.log("TFIDF:", tfidf);
    if (tfidf > 0) {
        sentenceRanks.push([sentence , tfidf]);
    }
}
sentenceRanks = sentenceRanks.sort((a, b) => b[1] - a[1]);

console.log(sentenceRanks);