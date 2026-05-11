(() => {
  const STORAGE_KEY = "research_assistant_session_id";

  const el = {
    messages: document.getElementById("messages"),
    clarifyBanner: document.getElementById("clarify-banner"),
    clarifyText: document.getElementById("clarify-text"),
    composer: document.getElementById("composer"),
    input: document.getElementById("input"),
    btnSend: document.getElementById("btn-send"),
    btnLabel: document.getElementById("btn-label"),
    btnSpinner: document.getElementById("btn-spinner"),
    errorBar: document.getElementById("error-bar"),
    btnNew: document.getElementById("btn-new-chat"),
  };

  let sessionId = localStorage.getItem(STORAGE_KEY);

  function showError(msg) {
    el.errorBar.textContent = msg;
    el.errorBar.classList.remove("hidden");
  }

  function hideError() {
    el.errorBar.classList.add("hidden");
  }

  function setLoading(loading) {
    el.btnSend.disabled = loading;
    el.input.disabled = loading;
    el.btnLabel.classList.toggle("hidden", loading);
    el.btnSpinner.classList.toggle("hidden", !loading);
  }

  function ensureSessionId() {
    if (!sessionId || sessionId.length < 8) {
      sessionId = crypto.randomUUID().replace(/-/g, "").slice(0, 24);
      localStorage.setItem(STORAGE_KEY, sessionId);
    }
  }

  async function newSession() {
    try {
      const r = await fetch("/api/session/new", { method: "POST" });
      const data = await r.json();
      sessionId = data.session_id;
      localStorage.setItem(STORAGE_KEY, sessionId);
      el.messages.innerHTML = "";
      el.clarifyBanner.classList.add("hidden");
      hideError();
    } catch (e) {
      showError("Could not start session: " + e.message);
    }
  }

  function appendBubble(role, text) {
    const div = document.createElement("div");
    div.className = "msg " + role;
    if (role === "assistant") {
      div.innerHTML = simpleMarkdown(text);
    } else {
      div.textContent = text;
    }
    el.messages.appendChild(div);
    el.messages.scrollTop = el.messages.scrollHeight;
  }

  /** Very small Markdown subset for ##, -, ** */
  function simpleMarkdown(raw) {
    let t = escapeHtml(raw);
    t = t.replace(/^### (.*)$/gm, "<h3>$1</h3>");
    t = t.replace(/^## (.*)$/gm, "<h2>$1</h2>");
    t = t.replace(/^# (.*)$/gm, "<h1>$1</h1>");
    t = t.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    t = t.replace(/^- (.+)$/gm, "<li>$1</li>");
    t = t.replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>");
    return t.replace(/\n/g, "<br />");
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  async function sendMessage() {
    const text = el.input.value.trim();
    if (!text) return;

    hideError();
    appendBubble("user", text);
    el.input.value = "";
    ensureSessionId();
    setLoading(true);
    el.clarifyBanner.classList.add("hidden");

    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });
      const data = await r.json();

      if (!data.ok || data.kind === "error") {
        showError(data.detail || "Request failed");
        return;
      }

      if (data.kind === "clarify") {
        el.clarifyText.textContent = data.question || "";
        el.clarifyBanner.classList.remove("hidden");
        el.messages.scrollTop = el.messages.scrollHeight;
        return;
      }

      if (data.kind === "reply") {
        appendBubble("assistant", data.text || "");
      }
    } catch (e) {
      showError(String(e.message || e));
    } finally {
      setLoading(false);
    }
  }

  el.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage();
  });

  el.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  el.btnNew.addEventListener("click", () => {
    if (confirm("Start a new chat? This clears the screen (server session is reset).")) {
      newSession();
    }
  });

  ensureSessionId();
})();
