// Déclare window globalement sur chrome (sinon problème dans les bundles)
if (typeof browser === "undefined") {
    globalThis.window = globalThis.window || globalThis;
}

import * as tf from '@tensorflow/tfjs';
/* import * as use from '@tensorflow-models/universal-sentence-encoder'; */

/* const browserAPI_serviceworker = typeof browser !== "undefined" ? browser : chrome;
let model;
let modelLoadingPromise = null; */
let trained_model;
const embeddingCache = new Map(); // Cache pour les embedding

async function loadTrainedModel() {
    try {
        // Tente de charger le modèle
        trained_model = await tf.loadLayersModel('model_tfjs/model.json');
        console.log('trained_model = await tf.loadLayersModel(model_tfjs/model.json);')
        return trained_model
    } catch (error) {
        // Si une erreur se produit, elle est capturée ici
        console.error('Erreur lors du chargement du modèle :', error);
    }
}

async function getEmbeddingsTrained(texts) {

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
        console.log(trained_model)
        debugger
        const embeddings = await trained_model.predict(textsToEmbed); // Utilisez la méthode appropriée ici
        const embeddingsArray = embeddings.arraySync();

        // Ajouter les nouveaux embeddings au cache et à la liste des résultats
        textsToEmbed.forEach((text, index) => {
            embeddingCache.set(text, embeddingsArray[index]);
            newEmbeddings.push(embeddingsArray[index]);
        });
    }

    return newEmbeddings;
}

// Fonction pour obtenir des prédictions avec ton modèle personnalisé
async function getPredictionWithTrainedModel(texts) {
    
    const embeddings = await getEmbeddingsTrained(texts); // Obtenir les embeddings des textes
    console.log('const embeddings = await getEmbeddingsTrained(texts);')
    const tensors = embeddings.map(embed => tf.tensor(embed)); // Convertir les embeddings en tenseurs
    console.log('const tensors = embeddings.map(embed => tf.tensor(embed));')
    // Faire des prédictions avec ton modèle personnalisé
    const predictions = trained_model.predict(tf.stack(tensors)); // Prédire avec le modèle
    console.log('const predictions = trained_model.predict(tf.stack(tensors));')
    console.log('return predictions.arraySync();')
    return predictions.arraySync(); // Retourner les résultats de la prédiction
}

// Exemple d'utilisation
async function analyzeText(texts) {
    const predictions = await getPredictionWithTrainedModel(texts);
    console.log("Prédictions : ", predictions);
}

// Charger le modèle dès le début
await loadTrainedModel().then(() => {
    console.log("Modèle chargé et prêt !");
});


analyzeText("Hello world!");


// Calculer la similarité cosinus entre deux vecteurs
/* function cosineSimilarity(vecA, vecB) {
    console.log('vecA.length', vecA.length)
    console.log('vecB.length', vecB.length)
    const dotProduct = vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
} */

/* async function analyzeText(sentence, keywords, threshold = 0.7) {
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
} */

/* function handleMessage(message, sender, sendResponse) {
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
browserAPI_serviceworker.runtime.onMessage.addListener(handleMessage); */


/* async function getEmbeddings(texts) {
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
} */
/* async function loadModel() {
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
} */