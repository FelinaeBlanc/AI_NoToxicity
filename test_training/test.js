(async function () {
    const charEncoder = (inputString, maxLength = 200) => {
        const normalizedString = inputString.toLowerCase().replace(/[^a-z ]/g, '');
        const charToIndex = char => {
            if (char === ' ') return 27; // Space
            const charCode = char.charCodeAt(0);
            return charCode >= 97 && charCode <= 122 ? charCode - 96 : 0; // a=1, ..., z=26
        };
        const encoded = normalizedString
            .split('')
            .map(charToIndex)
            .slice(0, maxLength);
        const padded = Array(maxLength).fill(0);
        encoded.forEach((val, idx) => padded[idx] = val);
        return tf.tensor2d([padded], [1, maxLength]);
    };

    function tokenizeText(text, wordIndex) {
        const words = text.toLowerCase().split(/\s+/); // Split text into words
        const sequence = words.map(word => wordIndex[word] || 0); // Map words to indices
        if (sequence.length > maxLen) {
            sequence.splice(0, sequence.length - maxLen); // Truncate sequence if too long
        }
        const paddedSequence = Array(maxLen).fill(0);
        for (let i = 0; i < sequence.length; i++) {
            paddedSequence[maxLen - sequence.length + i] = sequence[i]; // Add padding
        }
        return tf.tensor([paddedSequence]);
    }

    // Prédire depuis une phrase
    const predictFromText = async (inputString, model) => {
        const inputTensor = charEncoder(inputString);  // Change to word-level encoding if needed
        const prediction = model.predict(inputTensor);
        const result = (await prediction.array())[0][0];  // Assuming a single output for binary classification
        inputTensor.dispose();
        console.log('result:', inputString, result);
        return result > 0.5 ? 'Toxic' : 'Non-Toxic';
    };

    // Exemple d'utilisation
    const main = async () => {
        const model = await tf.loadLayersModel('./../model_tfjs/model.json');
        let inputString = "Hi how are you ?";
        let result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);
        
        // Test more sentences
        inputString = "stupid motherfucker";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "suck my dick";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "kill yourself";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "your mom is a whore";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "your mom is a bitch";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "your such a piece of shit";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "stupid";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "motherfucker";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "fucking bitch";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "fucking";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "bitch";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "I rawdogged your mom last night";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);

        // Test more sentences
        inputString = "you're such a pussy";
        result = await predictFromText(inputString, model);
        console.log('Prediction:', inputString, result);
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