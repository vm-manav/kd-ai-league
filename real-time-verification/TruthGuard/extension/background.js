chrome.runtime.onInstalled.addListener(() => {
    chrome.contextMenus.create({
        id: "verify-truthguard",
        title: "Verify with TruthGuard AI",
        contexts: ["selection"]
    });
});

// Listener not strictly needed for popup flow, but good for future features
chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === "verify-truthguard") {
        // Just alerts user for now, as opening popup programmatically is restricted in Manifest V3
        chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => alert("Please click the TruthGuard icon in your toolbar to view the report!")
        });
    }
});