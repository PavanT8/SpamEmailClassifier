document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const emailText = document.getElementById('emailText');
    const classifyBtn = document.getElementById('classifyBtn');
    const btnText = document.getElementById('btnText');
    const clearBtn = document.getElementById('clear-btn');
    const pasteBtn = document.getElementById('paste-btn');
    const charCountEl = document.getElementById('char-count');
    const wordCountEl = document.getElementById('word-count');

    // Preset Buttons
    const sampleSpamBtn = document.getElementById('sample-spam-btn');
    const samplePhishBtn = document.getElementById('sample-phish-btn');
    const sampleHamBtn = document.getElementById('sample-ham-btn');
    const sampleCasualBtn = document.getElementById('sample-casual-btn');

    // Result Elements
    const resultBox = document.getElementById('resultBox');
    const verdictBanner = document.getElementById('verdict-banner');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictTag = document.getElementById('verdict-tag');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictSummary = document.getElementById('verdict-summary');
    const riskBadge = document.getElementById('risk-badge');
    const spamScoreText = document.getElementById('spam-score-text');
    const spamMeterBar = document.getElementById('spam-meter-bar');
    const hamProbText = document.getElementById('ham-prob-text');
    const spamProbText = document.getElementById('spam-prob-text');
    const triggersList = document.getElementById('triggers-list');
    const copyResultBtn = document.getElementById('copy-result-btn');
    const resetResultBtn = document.getElementById('reset-result-btn');

    // Error Elements
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');

    // Navigation & Status
    const systemStatus = document.getElementById('system-status');
    const navAnalyzer = document.getElementById('nav-analyzer');
    const navAbout = document.getElementById('nav-about');

    // History Elements
    const historyList = document.getElementById('history-list');
    const clearHistoryBtn = document.getElementById('clear-history-btn');

    const samplePresets = {
        spam: "CONGRATULATIONS! You have been selected as the grand prize winner of $1,000,000 cash in our international sweepstakes! Call 0800 123 4567 immediately or reply CLAIM to collect your prize. Guaranteed payout, terms apply. Valid for 24 hours only!",
        phish: "Urgent Security Notice: Your online banking account access has been temporarily suspended due to multiple unauthorized login attempts. Click here immediately to verify your identity and restore your credentials: http://secure-bank-login-update.com/verify. Failure to do so within 12 hours will result in permanent account deactivation.",
        ham: "Hi team, please find attached the updated slide deck for tomorrow's Q3 performance review meeting scheduled at 10:00 AM in Conference Room B. Please review the slides beforehand and bring any feedback.",
        casual: "Hey Sarah, are we still meeting up for coffee this afternoon? Let me know what time works best for you. I should finish my lectures around 3:30 PM. See you soon!"
    };

    let latestAnalysis = null;

    // Check health on load
    async function checkHealth() {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                const data = await res.json();
                if (data.status === 'healthy') {
                    systemStatus.innerHTML = `<span class="status-dot"></span><span class="status-text">Model: Online (97.3% Acc)</span>`;
                }
            }
        } catch (e) {
            console.warn('Health check unavailable:', e);
        }
    }

    // Preset Listeners
    if (sampleSpamBtn) sampleSpamBtn.addEventListener('click', () => loadText(samplePresets.spam));
    if (samplePhishBtn) samplePhishBtn.addEventListener('click', () => loadText(samplePresets.phish));
    if (sampleHamBtn) sampleHamBtn.addEventListener('click', () => loadText(samplePresets.ham));
    if (sampleCasualBtn) sampleCasualBtn.addEventListener('click', () => loadText(samplePresets.casual));

    function loadText(text) {
        emailText.value = text;
        updateTextMetrics();
        hideResult();
        hideError();
        emailText.focus();
    }

    function updateTextMetrics() {
        const text = emailText.value;
        const chars = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        charCountEl.textContent = `${chars.toLocaleString()} character${chars === 1 ? '' : 's'}`;
        wordCountEl.textContent = `${words.toLocaleString()} word${words === 1 ? '' : 's'}`;
    }

    emailText.addEventListener('input', updateTextMetrics);

    clearBtn.addEventListener('click', () => {
        emailText.value = '';
        updateTextMetrics();
        hideResult();
        hideError();
        emailText.focus();
    });

    pasteBtn.addEventListener('click', async () => {
        try {
            if (navigator.clipboard && navigator.clipboard.readText) {
                const text = await navigator.clipboard.readText();
                if (text) {
                    emailText.value = text;
                    updateTextMetrics();
                    hideError();
                }
            } else {
                showError("Clipboard access not supported. Please paste with Ctrl+V.");
            }
        } catch (err) {
            showError("Unable to access clipboard. Please paste with Ctrl+V.");
        }
    });

    // Classification Request
    classifyBtn.addEventListener('click', async () => {
        const text = emailText.value.trim();

        if (!text) {
            showError("Please enter or paste email content before analyzing.");
            emailText.focus();
            return;
        }

        hideResult();
        hideError();
        setLoading(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => null);
                throw new Error(errData?.detail || errData?.error || `Server error (${response.status})`);
            }

            const data = await response.json();
            latestAnalysis = data;
            displayResult(data, text);
            saveToHistory(text, data);

        } catch (err) {
            console.error('Classification error:', err);
            showError(err.message || "Failed to analyze email. Please ensure the server is running.");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            classifyBtn.disabled = true;
            classifyBtn.innerHTML = '<span class="loader-spinner"></span><span>Classifying...</span>';
        } else {
            classifyBtn.disabled = false;
            classifyBtn.innerHTML = `
                <span class="btn-icon">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="11" cy="11" r="8"></circle>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                    </svg>
                </span>
                <span id="btnText">Classify Email</span>
            `;
        }
    }

    function displayResult(data, text) {
        resultBox.classList.remove('hidden');

        const isSpam = (data.prediction || '').toLowerCase() === 'spam';
        const spamProb = data.probabilities ? data.probabilities.spam : (isSpam ? data.confidence : (1 - data.confidence));
        const hamProb = data.probabilities ? data.probabilities.ham : (1 - spamProb);
        const spamPercent = Math.round(spamProb * 100);
        const hamPercent = Math.round(hamProb * 100);

        // Banner class
        verdictBanner.className = 'verdict-card ' + (isSpam ? 'is-spam' : 'is-ham');

        if (isSpam) {
            verdictIcon.innerHTML = `
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
            `;
            verdictTag.textContent = 'SECURITY ALERT';
            verdictTitle.textContent = 'SPAM / PHISHING DETECTED';
            verdictSummary.textContent = 'This message exhibits strong characteristics of unsolicited spam, promotional marketing, or phishing fraud.';
        } else {
            verdictIcon.innerHTML = `
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
                    <path d="m9 12 2 2 4-4"></path>
                </svg>
            `;
            verdictTag.textContent = 'SECURITY VERIFIED';
            verdictTitle.textContent = 'LEGITIMATE EMAIL (HAM)';
            verdictSummary.textContent = 'This message exhibits typical characteristics of authentic, safe personal or professional communication.';
        }

        // Risk badge
        const risk = data.risk_level || (isSpam ? 'High Threat' : 'Safe');
        riskBadge.textContent = risk;
        riskBadge.className = 'risk-badge ' + (
            risk === 'High Threat' ? 'risk-high' : 
            risk === 'Suspicious' || risk === 'Moderate' ? 'risk-moderate' : 'risk-safe'
        );

        // Probability meter
        spamScoreText.textContent = `${spamPercent}% Spam Score`;
        spamMeterBar.className = 'meter-bar ' + (isSpam ? 'spam-bar' : 'ham-bar');
        spamMeterBar.style.width = `${spamPercent}%`;

        hamProbText.textContent = `${hamPercent}%`;
        spamProbText.textContent = `${spamPercent}%`;

        // Triggers
        triggersList.innerHTML = '';
        const keywords = data.trigger_keywords || data.detected_keywords || [];
        if (keywords.length > 0) {
            keywords.forEach(word => {
                const tag = document.createElement('span');
                tag.className = 'trigger-tag';
                tag.textContent = word;
                triggersList.appendChild(tag);
            });
        } else {
            const noTriggers = document.createElement('span');
            noTriggers.className = 'no-triggers';
            noTriggers.textContent = isSpam 
                ? 'Flagged based on underlying statistical feature distributions.' 
                : 'No suspicious trigger keywords detected.';
            triggersList.appendChild(noTriggers);
        }

        resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function hideResult() { resultBox.classList.add('hidden'); }
    function showError(msg) {
        errorContainer.classList.remove('hidden');
        errorMessage.textContent = msg;
        errorContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    function hideError() { errorContainer.classList.add('hidden'); }

    // Copy Analysis
    copyResultBtn.addEventListener('click', async () => {
        if (!latestAnalysis) return;

        const summary = [
            `--- SpamGuard AI Analysis ---`,
            `Prediction: ${latestAnalysis.prediction.toUpperCase()}`,
            `Confidence: ${(latestAnalysis.confidence * 100).toFixed(1)}%`,
            `Risk Level: ${latestAnalysis.risk_level}`,
            `Spam Probability: ${Math.round(latestAnalysis.probabilities?.spam * 100 || 0)}%`,
            `Ham Probability: ${Math.round(latestAnalysis.probabilities?.ham * 100 || 0)}%`,
            `Trigger Keywords: ${(latestAnalysis.trigger_keywords || latestAnalysis.detected_keywords)?.join(', ') || 'None'}`,
            `Repository: https://github.com/PavanT8/SpamEmailClassifier`
        ].join('\n');

        try {
            await navigator.clipboard.writeText(summary);
            const original = copyResultBtn.innerHTML;
            copyResultBtn.innerHTML = `✓ Copied!`;
            setTimeout(() => { copyResultBtn.innerHTML = original; }, 2000);
        } catch (err) {
            alert("Analysis copied:\n\n" + summary);
        }
    });

    resetResultBtn.addEventListener('click', () => {
        emailText.value = '';
        updateTextMetrics();
        hideResult();
        emailText.focus();
    });

    // History Persistence
    const HISTORY_KEY = 'spam_classifier_v2_history';

    function getHistory() {
        try {
            const stored = localStorage.getItem(HISTORY_KEY);
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            return [];
        }
    }

    function saveToHistory(text, result) {
        try {
            const history = getHistory();
            const newItem = {
                id: Date.now(),
                text: text,
                prediction: result.prediction,
                confidence: result.confidence,
                risk: result.risk_level,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            };
            history.unshift(newItem);
            if (history.length > 5) history.pop();
            localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
            renderHistory();
        } catch (e) {
            console.warn('History storage failed:', e);
        }
    }

    function renderHistory() {
        const history = getHistory();
        historyList.innerHTML = '';

        if (history.length === 0) {
            historyList.innerHTML = `<p class="history-empty">No messages analyzed yet in this session. Try clicking a preset above or paste an email.</p>`;
            return;
        }

        history.forEach(item => {
            const el = document.createElement('div');
            el.className = 'history-item';
            const isSpam = item.prediction.toLowerCase() === 'spam';
            el.innerHTML = `
                <div class="history-item-left">
                    <span class="history-badge ${isSpam ? 'spam' : 'ham'}">${isSpam ? 'Spam' : 'Ham'}</span>
                    <span class="history-text" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</span>
                </div>
                <div class="history-meta">
                    <span>${Math.round(item.confidence * 100)}%</span>
                    <span>${item.timestamp}</span>
                </div>
            `;
            el.addEventListener('click', () => loadText(item.text));
            historyList.appendChild(el);
        });
    }

    clearHistoryBtn.addEventListener('click', () => {
        localStorage.removeItem(HISTORY_KEY);
        renderHistory();
    });

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Initialize
    checkHealth();
    renderHistory();
    updateTextMetrics();
});
