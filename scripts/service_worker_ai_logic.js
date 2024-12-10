// Déclare window globalement sur chrome (sinon problème dans les bundles)
if (typeof browser === "undefined") {
    globalThis.window = globalThis.window || globalThis;
}

import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';

const browserAPI_serviceworker = typeof browser !== "undefined" ? browser : chrome;
let model;
let modelLoadingPromise = null;
let trained_model;
const embeddingCache = new Map(); // Cache pour les embedding

async function loadModel() {
    if (!model) {
        if (!modelLoadingPromise) {
            modelLoadingPromise = use.load().then(loadedModel => {
                model = loadedModel;
                modelLoadingPromise = null;
                console.log("Modèle chargé !");

                return model;
            }).catch(error => {
                modelLoadingPromise = null;
                throw error;
            });
        }
        return modelLoadingPromise;
    }
    return model;
}

async function loadTrainedModel() {
    try {
        // Tente de charger le modèle
        trained_model = await tf.loadLayersModel('model_tfjs/model.json');
        console.log("trained_model loaded successfully!");
        print('model.predict(gros fils de pute)', model.predict('gros fils de pute'))

        return model
    } catch (error) {
        // Si une erreur se produit, elle est capturée ici
        console.error('Erreur lors du chargement du modèle :', error);
        alert("Une erreur est survenue lors du chargement du modèle. Veuillez vérifier le chemin ou la connexion.");
    }
}

async function getEmbeddings(texts) {
    await loadModel(); // Vérifie que le modèle est chargé

    const newEmbeddings = [];
    const textsToEmbed = [];

    // Vérifie le cache pour chaque texte
    for (const text of texts) {
        if (embeddingCache.has(text)) {
            newEmbeddings.push(embeddingCache.get(text));
        } else {
            textsToEmbed.push(text);
        }
    }

    // Obtenir les embeddings pour les nouveaux textes
    if (textsToEmbed.length > 0) {
        const embeddings = await model.embed(textsToEmbed);
        const embeddingsArray = embeddings.arraySync();

        // Ajouter les nouveaux embeddings au cache et à la liste des résultats
        textsToEmbed.forEach((text, index) => {
            embeddingCache.set(text, embeddingsArray[index]);
            newEmbeddings.push(embeddingsArray[index]);
        });
    }

    return newEmbeddings; // retourne un tableau 2D du vecteur
}

async function getEmbeddingsTrained(texts) {
    await loadModel(true); // Charger le modèle personnalisé TF.js

    const newEmbeddings = [];
    const textsToEmbed = [];

    // Vérifie le cache pour chaque texte
    for (const text of texts) {
        if (embeddingCache.has(text)) {
            newEmbeddings.push(embeddingCache.get(text));
        } else {
            textsToEmbed.push(text);
        }
    }

    // Transformer les textes en tenseurs pour les prédictions
    if (textsToEmbed.length > 0) {
        const tensors = textsToEmbed.map(text => tf.tensor(text.split(' '))); // Exemple de traitement simple
        const embeddings = await model.predict(tensors); // Utilisez la méthode appropriée ici
        const embeddingsArray = embeddings.arraySync();

        // Ajouter les nouveaux embeddings au cache et à la liste des résultats
        textsToEmbed.forEach((text, index) => {
            embeddingCache.set(text, embeddingsArray[index]);
            newEmbeddings.push(embeddingsArray[index]);
        });
    }

    return newEmbeddings;
}


// Calculer la similarité cosinus entre deux vecteurs
function cosineSimilarity(vecA, vecB) {
    console.log('vecA.length', vecA.length)
    console.log('vecB.length', vecB.length)
    const dotProduct = vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}

async function analyzeText(sentence, keywords, threshold = 0.7) {
    try {
        // Obtenir les embeddings pour tous les mots en une seule fois
        //const texts = [sentence, ...keywords];
        const embeddingsA = await getEmbeddingsTrained([sentence]);
        const embeddingsB = await getEmbeddingsTrained(keywords);

        console.log("OKi embedding")
        // Séparer l'embedding de la phrase et des mots-clés
        const sentenceEmbedding = embeddingsA[0];
        const keywordEmbeddings = embeddingsB;

        console.log("wordEmbeddings", sentence, sentenceEmbedding);
        console.log("keywordEmbeddings", keywords, keywordEmbeddings);
        
        let keywordCpt = 0;
        for (const keywordEmbedding of keywordEmbeddings) {

            const similarity = cosineSimilarity(sentenceEmbedding, keywordEmbedding);
            console.log("Similarité avec", keywords[keywordCpt], sentence, ":", similarity);
            if (similarity > threshold) {
                return true; // Retourne true dès qu'un mot dépasse le seuil
            }
            keywordCpt++;
        }
        return false; // Retourne false si aucun mot ne dépasse le seuil
    } catch (error) {
        console.log(error);
        return false;
    }
}


loadTrainedModel(); // On charge le modèle dès le début

function handleMessage(message, sender, sendResponse) {
    if (message.type === 'analyzeText' && message.sentence && Array.isArray(message.keywords)) {
        console.log("Handle MEssageeeeeeeeeeeeeeeeeeeeeeeee!!! => "+message.sentence)
        
        analyzeText(message.sentence, message.keywords, message.threshold)
            .then(result => {
                sendResponse({ isAboveThreshold: result });
            })
            .catch(error => {
                console.error(error);
                sendResponse({ isAboveThreshold: false });
            });
        return true; // Indique une réponse asynchrone à l’API
    }
}
browserAPI_serviceworker.runtime.onMessage.addListener(handleMessage);