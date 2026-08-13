document.getElementById('classificationForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const emailText = document.getElementById('emailText').value;
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const loader = document.getElementById('loader');
    const resultBox = document.getElementById('resultBox');
    const resultContent = document.getElementById('resultContent');

    if (!emailText.trim()) return;

    // UI Loading state
    submitBtn.disabled = true;
    btnText.textContent = 'Classifying...';
    loader.classList.remove('hidden');
    resultBox.classList.add('hidden');
    resultContent.className = 'result-content'; // reset classes

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: emailText })
        });

        const data = await response.json();

        if (response.ok) {
            const prediction = data.prediction.toLowerCase();
            resultContent.textContent = prediction === 'spam' ? 'SPAM 🚫' : 'HAM (Not Spam) ✅';
            resultContent.classList.add(`result-${prediction}`);
        } else {
            resultContent.textContent = data.error || 'An error occurred during classification.';
            resultContent.classList.add('result-error');
        }
    } catch (error) {
        console.error('Error:', error);
        resultContent.textContent = 'Failed to connect to the server.';
        resultContent.classList.add('result-error');
    } finally {
        // Reset UI state
        submitBtn.disabled = false;
        btnText.textContent = 'Classify Email';
        loader.classList.add('hidden');
        resultBox.classList.remove('hidden');
    }
});
