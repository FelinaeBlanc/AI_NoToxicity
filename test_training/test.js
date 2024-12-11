(async function () {
    const charEncoder = (inputString, maxLength = 200) => {
        const normalizedString = inputString.toLowerCase().replace(/[^a-z ]/g, '');
        const charToIndex = char => {
            if (char === ' ') return 27; // espace
            const charCode = char.charCodeAt(0);
            return charCode >= 97 && charCode <= 122 ? charCode - 96 : 0; // a=1, ..., z=26
        };
        const encoded = normalizedString
            .split('')
            .map(charToIndex)
            .slice(0, maxLength);
        const padded = Array(maxLength).fill(0);
        encoded.forEach((val, idx) => {
            padded[idx] = val;
        });
        return tf.tensor2d([padded], [1, maxLength]);
    };

    // Prédire depuis une phrase
    const predictFromText = async (inputString, model) => {
        const inputTensor = charEncoder(inputString);
        const prediction = model.predict(inputTensor);
        const result = (await prediction.array())[0][0];
        inputTensor.dispose();
        console.log(await prediction.array())
        debugger
        return result > 0.5 ? 'Positive' : 'Negative';
    };

    // Exemple d'utilisation
    const main = async () => {
        const model = await tf.loadLayersModel('./../model_tfjs/model.json');
        let inputString = "xddddddddddddddddd";
        let result = await predictFromText(inputString, model);
        console.log('Prediction:', result);

        inputString = "stupid motherfucker";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', result);

        inputString = "you're such a piece of shit lmao";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', result);

        inputString = "go kill yourself dumbfuck";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', result);
    };

    main();
    
})();


// var trained_model;
    // async function loadTrainedModel() {
    //     try {
    //         // Tente de charger le modèle
    //         console.log(tf);
    //         trained_model = await tf.loadLayersModel('./../model_tfjs/model.json');
    //         console.log('trained_model = await tf.loadLayersModel(model_tfjs/model.json);')
    //         return trained_model
    //     } catch (error) {
    //         // Si une erreur se produit, elle est capturée ici
    //         console.error('Erreur lors du chargement du modèle :', error);
    //     }

    // }

    // const tokenizeInput = (inputString, wordIndex, maxLength = 200) => {
    //     const words = inputString.toLowerCase().split(/\s+/); // Découpe par mots
    //     const tokenized = words.map(word => wordIndex[word] || 0); // Associe les mots à leurs indices
    //     const padded = Array(maxLength).fill(0); // Ajoute du padding si nécessaire
    //     tokenized.slice(0, maxLength).forEach((value, index) => {
    //         padded[index] = value;
    //     });
    //     return tf.tensor2d([padded], [1, maxLength]); // Forme [1, maxLength]
    // };
    
    // await loadTrainedModel()
    // console.log(trained_model)

    // Fonction d'encodage