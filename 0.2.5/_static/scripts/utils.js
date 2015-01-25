function getSnippet(id, url) {
    var req = false;
    // For Safari, Firefox, and other non-MS browsers
    if (window.XMLHttpRequest) {
        try {
            req = new XMLHttpRequest();
        } catch (e) {
            req = false;
        }
    } else if (window.ActiveXObject) {
        // For Internet Explorer on Windows
        try {
            req = new ActiveXObject("Msxml2.XMLHTTP");
        } catch (e) {
            try {
                req = new ActiveXObject("Microsoft.XMLHTTP");
            } catch (e) {
                req = false;
            }
        }
    }
    var element = document.getElementById(id);
    if (req) {
        // Synchronous request, wait till we have it all
        req.open('GET', url, false);
        req.send(null);
        element.innerHTML = req.responseText;
    }
}
