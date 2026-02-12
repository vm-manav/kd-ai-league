chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "getSelection") {
        const selectedText = window.getSelection().toString().trim();
        sendResponse({ selection: selectedText });
    }
});