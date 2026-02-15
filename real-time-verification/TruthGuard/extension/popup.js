document.addEventListener('DOMContentLoaded', () => {
    const claimTextElem = document.getElementById('claim-text');
    const claimInput = document.getElementById('claim-input');
    const verifyBtn = document.getElementById('verify-btn');
    const loader = document.getElementById('loader');
    const resultArea = document.getElementById('result-area');

    function updateVerifyButton() {
        const typed = (claimInput.value || '').trim();
        const selected = (claimTextElem.innerText || '').trim();
        const hasSelection = selected.length > 0 && selected !== 'No text selected on page.';
        verifyBtn.disabled = typed.length === 0 && !hasSelection;
    }

    // 1. Get Selected Text from Browser (show in "selected" area; optional pre-fill)
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
        if (!tabs[0] || !tabs[0].id) {
            updateVerifyButton();
            return;
        }
        chrome.tabs.sendMessage(tabs[0].id, {action: "getSelection"}, (response) => {
            if (response && response.selection && response.selection.length > 0) {
                claimTextElem.innerText = response.selection;
                if (!claimInput.value.trim()) claimInput.placeholder = "Or type a different claim...";
            } else {
                claimTextElem.innerText = "No text selected on page.";
            }
            updateVerifyButton();
        });
    });

    claimInput.addEventListener('input', updateVerifyButton);
    claimInput.addEventListener('paste', () => setTimeout(updateVerifyButton, 0));

    // 2. Handle Verify Button Click — use typed text if any, else selected text
    verifyBtn.addEventListener('click', async () => {
        const typed = claimInput.value.trim();
        const selected = claimTextElem.innerText.trim();
        const textToVerify = typed.length > 0 ? typed : (selected && selected !== "No text selected on page." ? selected : '');
        
        // Show Loader
        loader.classList.remove('hidden');
        resultArea.classList.add('hidden');
        verifyBtn.disabled = true;

        if (!textToVerify) {
            alert("Enter or select a claim to verify.");
            return;
        }

        try {
            // Call Python Backend
            const response = await fetch('http://localhost:8000/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: textToVerify })
            });

            if (!response.ok) throw new Error('Server Error');
            const data = await response.json();
            
            displayResult(data);

        } catch (error) {
            alert("Connection Failed: Ensure backend is running at localhost:8000");
            console.error(error);
        } finally {
            loader.classList.add('hidden');
            verifyBtn.disabled = false;
        }
    });
});

function displayResult(data) {
    const resultArea = document.getElementById('result-area');
    const verdictBox = document.getElementById('verdict-box');
    const reasoningText = document.getElementById('reasoning-text');
    const sourcesList = document.getElementById('sources-list');

    resultArea.classList.remove('hidden');

    // Set Verdict Logic
    verdictBox.innerText = data.verdict.toUpperCase();
    verdictBox.className = ''; // Reset classes
    
    const v = data.verdict.toLowerCase();
    if (v.includes('true')) verdictBox.classList.add('v-true');
    else if (v.includes('false')) verdictBox.classList.add('v-false');
    else if (v.includes('misleading')) verdictBox.classList.add('v-misleading');
    else if (v.includes('not a factual') || v.includes('not factual')) verdictBox.classList.add('v-not-factual');
    else verdictBox.classList.add('v-unverified');

    // Set Text
    reasoningText.innerText = data.reasoning || '';

    // Set Sources (guard against missing array)
    sourcesList.innerHTML = '';
    (data.sources || []).forEach(src => {
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = src.url || '#';
        a.target = '_blank';
        a.rel = 'noopener noreferrer';
        const span = document.createElement('span');
        span.textContent = src.title || 'Source';
        a.appendChild(span);
        li.appendChild(a);
        sourcesList.appendChild(li);
    });
}