"use strict";

const elements = {
  shell: document.querySelector("#app-shell"),
  sidebar: document.querySelector("#sidebar"),
  sidebarScrim: document.querySelector("#sidebar-scrim"),
  openSidebar: document.querySelector("#open-sidebar"),
  sidebarToggle: document.querySelector("#sidebar-toggle"),
  appIcon: document.querySelector("#app-icon"),
  appTitle: document.querySelector("#app-title"),
  newChat: document.querySelector("#new-chat"),
  conversationList: document.querySelector("#conversation-list"),
  modelName: document.querySelector("#model-name"),
  systemBanner: document.querySelector("#system-banner"),
  systemBannerText: document.querySelector("#system-banner-text"),
  messageStage: document.querySelector("#message-stage"),
  messages: document.querySelector("#messages"),
  jumpLatest: document.querySelector("#jump-latest"),
  composerZone: document.querySelector("#composer-zone"),
  composerInput: document.querySelector("#composer-input"),
  fileInput: document.querySelector("#file-input"),
  attachButton: document.querySelector("#attach-button"),
  attachmentTray: document.querySelector("#attachment-tray"),
  contextUsage: document.querySelector("#context-usage"),
  sendReason: document.querySelector("#send-reason"),
  sendButton: document.querySelector("#send-button"),
  stopButton: document.querySelector("#stop-button"),
  toastRegion: document.querySelector("#toast-region"),
  streamStatus: document.querySelector("#stream-status"),
  deleteConfirm: document.querySelector("#delete-confirm"),
  deleteConfirmCancel: document.querySelector("#delete-confirm-cancel"),
  deleteConfirmSubmit: document.querySelector("#delete-confirm-submit"),
};

const state = {
  config: null,
  conversations: [],
  activeId: null,
  messages: [],
  pendingFiles: [],
  generating: false,
  controller: null,
  nearBottom: true,
  composing: false,
  renderFrame: null,
  dragDepth: 0,
  aiIconPreload: null,
};

const MUTATION_HEADERS = { "X-Simple-Chat-Request": "1" };
const formatter = new Intl.DateTimeFormat("ja-JP", {
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
});
const mobileViewport = window.matchMedia("(max-width: 820px)");

document.addEventListener("DOMContentLoaded", initialize);

async function initialize() {
  bindEvents();
  try {
    const initial = await apiJson("/api/state");
    state.config = initial.config;
    state.conversations = initial.conversations;
    applyConfig();
    renderConversationList();
    beginNewChat();
  } catch (error) {
    showToast(error.message || "アプリを初期化できませんでした。", "error", 8000);
    showBanner("ローカルサーバーへ接続できません。ページを再読み込みしてください。");
  }
}

function bindEvents() {
  elements.newChat.addEventListener("click", () => {
    beginNewChat();
  });
  elements.openSidebar.addEventListener("click", openSidebar);
  elements.sidebarToggle.addEventListener("click", toggleSidebar);
  elements.sidebarScrim.addEventListener("click", () => closeSidebar(true));
  elements.attachButton.addEventListener("click", () => elements.fileInput.click());
  elements.fileInput.addEventListener("change", (event) => {
    addFiles([...event.target.files]);
    event.target.value = "";
  });
  elements.composerInput.addEventListener("input", () => {
    resizeComposer();
    updateComposerState();
  });
  elements.composerInput.addEventListener("compositionstart", () => { state.composing = true; });
  elements.composerInput.addEventListener("compositionend", () => { state.composing = false; });
  elements.composerInput.addEventListener("keydown", handleComposerKeydown);
  elements.composerInput.addEventListener("paste", handlePaste);
  elements.sendButton.addEventListener("click", () => sendMessage());
  elements.stopButton.addEventListener("click", stopGeneration);
  elements.jumpLatest.addEventListener("click", () => scrollToBottom(true));
  elements.messageStage.addEventListener("scroll", handleMessageScroll, { passive: true });
  elements.messages.addEventListener("click", handleMessageAction);

  elements.composerZone.addEventListener("dragenter", handleDragEnter);
  elements.composerZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
  });
  elements.composerZone.addEventListener("dragleave", handleDragLeave);
  elements.composerZone.addEventListener("drop", handleDrop);

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.generating) stopGeneration();
    if (event.key === "Escape" && elements.shell.classList.contains("sidebar-open")) {
      closeSidebar(true);
    }
  });
  mobileViewport.addEventListener("change", handleSidebarViewportChange);
  syncSidebarState();
}

function applyConfig() {
  document.title = state.config.app_title;
  elements.appIcon.href = state.config.ai_icon_url;
  elements.appTitle.textContent = state.config.app_title;
  elements.modelName.textContent = state.config.current_model_label;
  state.aiIconPreload = new Image();
  state.aiIconPreload.src = state.config.ai_icon_url;
  elements.composerInput.maxLength = state.config.max_text_length;
  elements.fileInput.accept = state.config.images_enabled ? "image/jpeg,image/png,image/webp" : "";
  if (!state.config.llm_configured) {
    showBanner("OPENAI_API_KEY が未設定です。履歴の閲覧と整理はできますが、新しいメッセージは送信できません。");
  }
}

function beginNewChat() {
  if (state.generating) return;
  state.activeId = null;
  state.messages = [];
  renderConversationList();
  renderDraftHeader();
  renderMessages({ forceBottom: true });
  updateComposerState();
  closeSidebar();
  elements.composerInput.focus();
}

async function refreshConversations() {
  const current = await apiJson("/api/state");
  state.conversations = current.conversations;
  renderConversationList();
  const active = activeConversation();
  if (active) renderActiveHeader(active);
}

async function selectConversation(conversationId) {
  if (state.generating && conversationId !== state.activeId) {
    showToast("回答生成中は別の会話へ移動できません。", "error");
    return;
  }
  state.activeId = conversationId;
  const detail = await apiJson(`/api/conversations/${conversationId}`);
  const { conversation, messages } = detail;
  const index = state.conversations.findIndex((item) => item.id === conversationId);
  if (index >= 0) state.conversations[index] = conversation;
  else state.conversations.unshift(conversation);
  state.messages = messages;
  renderConversationList();
  renderActiveHeader(conversation);
  renderMessages({ forceBottom: true });
  updateComposerState();
  closeSidebar();
}

function activeConversation() {
  return state.conversations.find((item) => item.id === state.activeId) || null;
}

function renderActiveHeader(conversation) {
  elements.modelName.textContent = conversation.model_label;
  renderContextUsage(conversation.context_tokens);
  if (!conversation.continuable) {
    showBanner("OpenAI側の会話コンテキストが失効したため、この会話は継続できません。新しいチャットを開始してください。");
  } else if (state.config.llm_configured) {
    elements.systemBanner.hidden = true;
  }
}

function renderDraftHeader() {
  elements.modelName.textContent = state.config.current_model_label;
  renderContextUsage(0);
  if (state.config.llm_configured) elements.systemBanner.hidden = true;
}

function renderContextUsage(tokens) {
  const value = Number(tokens) || 0;
  elements.contextUsage.hidden = value <= 0;
  elements.contextUsage.textContent = value > 0
    ? `Context ${value.toLocaleString("ja-JP")} tokens`
    : "";
}

function renderConversationList() {
  elements.conversationList.replaceChildren();
  for (const conversation of state.conversations) {
    const item = document.createElement("div");
    item.className = `conversation-item${conversation.id === state.activeId ? " active" : ""}`;

    const open = document.createElement("button");
    open.type = "button";
    open.className = "conversation-open";
    open.setAttribute("aria-label", `${conversation.title}を開く`);
    if (conversation.id === state.activeId) open.setAttribute("aria-current", "page");
    open.addEventListener("click", () => selectConversation(conversation.id).catch(reportError));

    const name = document.createElement("span");
    name.className = "conversation-name";
    name.textContent = conversation.title;
    const meta = document.createElement("span");
    meta.className = "conversation-meta";
    const model = document.createElement("span");
    model.textContent = conversation.model_label;
    const time = document.createElement("time");
    time.dateTime = conversation.updated_at;
    time.textContent = safeDate(conversation.updated_at);
    meta.append(model, time);
    open.append(name, meta);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "conversation-delete";
    remove.textContent = "×";
    remove.setAttribute("aria-label", "会話を削除");
    remove.addEventListener("click", () => deleteConversation(conversation.id, remove));
    item.append(open, remove);
    elements.conversationList.append(item);
  }
}

async function deleteConversation(conversationId, anchor) {
  if (!conversationId || state.generating) return;
  const conversation = state.conversations.find((item) => item.id === conversationId);
  if (!conversation) return;
  const accepted = await askDeleteConfirm(anchor);
  if (!accepted) return;
  try {
    await apiJson(`/api/conversations/${conversationId}`, { method: "DELETE" });
    state.conversations = state.conversations.filter((item) => item.id !== conversationId);
    if (state.activeId === conversationId) {
      state.activeId = null;
      state.messages = [];
      if (state.conversations.length) await selectConversation(state.conversations[0].id);
      else beginNewChat();
    } else {
      renderConversationList();
    }
    showToast("会話を削除しました。");
  } catch (error) {
    reportError(error);
  }
}

function renderMessages({ forceBottom = false } = {}) {
  const keepAtBottom = forceBottom || state.nearBottom;
  elements.messages.replaceChildren();
  for (const message of state.messages) {
    elements.messages.append(renderMessage(message));
  }
  if (keepAtBottom) requestAnimationFrame(() => scrollToBottom(false));
  else elements.jumpLatest.hidden = true;
}

function renderMessage(message) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;
  article.dataset.messageId = message.id;

  const content = document.createElement("div");
  content.className = "message-content";
  const body = document.createElement("div");
  body.className = "message-body";
  if (message.role === "assistant") renderAssistant(body, message.content, Boolean(message.pending));
  else body.textContent = message.content;

  if (message.attachments?.length) body.append(renderMessageImages(message.attachments));
  content.append(body);

  if (message.role === "assistant") article.append(aiAvatar(), content);
  else article.append(content, userAvatar());
  return article;
}

function aiAvatar() {
  const avatar = document.createElement("div");
  avatar.className = "ai-avatar";
  const image = document.createElement("img");
  image.src = state.config.ai_icon_url;
  image.alt = "AI";
  image.addEventListener("error", () => avatar.classList.add("image-error"), { once: true });
  avatar.append(image);
  return avatar;
}

function userAvatar() {
  const avatar = document.createElement("div");
  avatar.className = "user-avatar";
  avatar.setAttribute("aria-label", "あなた");
  const svg = svgIcon("0 0 24 24", "user-avatar-icon");
  svg.append(svgShape("path", {
    d: "M12 12a4.2 4.2 0 1 0 0-8.4 4.2 4.2 0 0 0 0 8.4Zm0 2c-4.4 0-8 2.3-8 5.2 0 .7.6 1.2 1.3 1.2h13.4c.7 0 1.3-.5 1.3-1.2 0-2.9-3.6-5.2-8-5.2Z",
  }));
  avatar.append(svg);
  return avatar;
}

function renderAssistant(container, markdown, streaming) {
  const unsafeHtml = window.marked.parse(markdown || "");
  const safeHtml = window.DOMPurify.sanitize(unsafeHtml, {
    USE_PROFILES: { html: true },
    FORBID_TAGS: ["style", "iframe", "object", "embed", "form", "input", "button", "textarea", "select"],
    FORBID_ATTR: ["style"],
  });
  container.innerHTML = safeHtml;
  for (const link of container.querySelectorAll("a")) {
    if (!isSafeLink(link.getAttribute("href") || "")) link.removeAttribute("href");
    else {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    }
  }
  if (!streaming) decorateCodeBlocks(container);
  if (streaming) {
    if (!markdown) {
      const placeholder = document.createElement("span");
      placeholder.className = "generation-placeholder";
      placeholder.textContent = "回答を生成中…";
      container.append(placeholder);
    }
    const cursor = document.createElement("span");
    cursor.className = "stream-cursor";
    cursor.setAttribute("aria-hidden", "true");
    container.append(cursor);
  }
}

function decorateCodeBlocks(container) {
  for (const pre of [...container.querySelectorAll("pre")]) {
    if (pre.parentElement?.classList.contains("code-frame")) continue;
    const code = pre.querySelector("code");
    if (!code) continue;
    try { window.hljs.highlightElement(code); } catch (_) { /* Plain code remains readable. */ }
    const languageClass = [...code.classList].find((value) => value.startsWith("language-"));
    const language = languageClass ? languageClass.slice(9) : "text";
    const frame = document.createElement("div");
    frame.className = "code-frame";
    const toolbar = document.createElement("div");
    toolbar.className = "code-toolbar";
    const label = document.createElement("span");
    label.className = "code-language";
    label.textContent = language;
    const wrap = codeAction("折り返し", "wrap");
    const copy = codeAction("コピー", "copy");
    toolbar.append(label, wrap, copy);
    pre.replaceWith(frame);
    frame.append(toolbar, pre);
  }
}

function codeAction(label, action) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "code-action";
  button.dataset.codeAction = action;
  if (action === "copy") {
    button.classList.add("icon-code-action");
    button.setAttribute("aria-label", "コードをコピー");
    button.title = "コードをコピー";
    const copy = svgIcon("0 0 24 24", "copy-icon");
    copy.append(
      svgShape("rect", { x: "8", y: "8", width: "11", height: "11", rx: "2" }),
      svgShape("path", { d: "M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" }),
    );
    const complete = svgIcon("0 0 24 24", "copy-complete");
    complete.append(svgShape("path", { d: "m5 12 4 4L19 6" }));
    button.append(copy, complete);
  } else {
    button.textContent = label;
  }
  return button;
}

function svgIcon(viewBox, className) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", viewBox);
  svg.setAttribute("aria-hidden", "true");
  if (className) svg.setAttribute("class", className);
  return svg;
}

function svgShape(name, attributes) {
  const shape = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) shape.setAttribute(key, value);
  return shape;
}

async function handleMessageAction(event) {
  const button = event.target.closest("[data-code-action]");
  if (!button) return;
  const frame = button.closest(".code-frame");
  const pre = frame?.querySelector("pre");
  if (!pre) return;
  if (button.dataset.codeAction === "wrap") {
    pre.classList.toggle("wrap");
    button.textContent = pre.classList.contains("wrap") ? "折返し解除" : "折り返し";
  } else if (button.dataset.codeAction === "copy") {
    try {
      await navigator.clipboard.writeText(pre.textContent || "");
      button.dataset.copied = "true";
      button.setAttribute("aria-label", "コピーしました");
      button.title = "コピーしました";
      setTimeout(() => {
        delete button.dataset.copied;
        button.setAttribute("aria-label", "コードをコピー");
        button.title = "コードをコピー";
      }, 1200);
    } catch (_) {
      showToast("コードをコピーできませんでした。", "error");
    }
  }
}

function renderMessageImages(attachments) {
  const grid = document.createElement("div");
  grid.className = "message-images";
  for (const attachment of attachments) {
    const link = document.createElement("a");
    link.className = "message-image-link";
    link.href = attachment.content_url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    const image = document.createElement("img");
    image.src = attachment.content_url;
    image.alt = attachment.original_name || "添付画像";
    image.loading = "lazy";
    link.append(image);
    grid.append(link);
  }
  return grid;
}

async function sendMessage() {
  if (state.generating) return;
  const reason = composerBlockReason();
  if (reason) {
    if (reason !== "本文または画像を入力してください。") showToast(reason, "error");
    return;
  }

  const text = elements.composerInput.value;
  const pending = [...state.pendingFiles];
  const form = new FormData();
  form.append("text", text);
  if (state.activeId) form.append("conversation_id", state.activeId);
  for (const item of pending) form.append("images", item.file, item.file.name || "pasted-image.png");

  const optimisticTurn = addOptimisticTurn(text, pending);
  setGenerating(true);
  state.controller = new AbortController();
  let started = false;
  let completedId = null;
  let failure = null;
  try {
    const response = await fetch("/api/messages", {
      method: "POST",
      headers: MUTATION_HEADERS,
      body: form,
      signal: state.controller.signal,
    });
    if (!response.ok) throw await errorFromResponse(response);
    if (!response.body) throw new Error("ストリーミング応答を読み取れません。", { cause: "stream" });
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/x-ndjson")) throw new Error("応答形式が正しくありません。");

    await readNdjson(response.body, async (event) => {
      if (completedId) throw new Error("終端イベントの後にデータを受信しました。");
      if (event.type === "start") {
        if (started) throw new Error("開始イベントが重複しています。");
        started = true;
      } else if (event.type === "delta") {
        if (!started) throw new Error("開始前に差分を受信しました。");
        appendStreamDelta(optimisticTurn.assistantMessage, event.text || "");
      } else if (event.type === "completed") {
        if (!started) throw new Error("開始前に完了しました。");
        completedId = event.conversation_id;
      } else if (event.type === "error") {
        if (!started) throw new Error("開始前にエラーイベントを受信しました。");
        const error = new Error(event.message || "回答生成に失敗しました。");
        error.code = event.code;
        throw error;
      } else {
        throw new Error("未知のストリームイベントを受信しました。");
      }
    });
    if (!started || !completedId) throw new Error("回答ストリームが途中で終了しました。");
  } catch (error) {
    failure = error;
  } finally {
    state.controller = null;
    try {
      if (completedId) await refreshAfterTurn(completedId);
      else restoreUnsentDraft(optimisticTurn, text, pending);
      if (failure?.name !== "AbortError" && failure) reportError(failure);
      if (failure?.code === "context_expired" && state.activeId) await refreshAfterTurn(state.activeId);
    } catch (error) {
      reportError(error);
    } finally {
      setGenerating(false);
    }
  }
}

function restoreUnsentDraft(optimisticTurn, text, pending) {
  state.messages = state.messages.filter(
    (message) => message !== optimisticTurn.userMessage && message !== optimisticTurn.assistantMessage,
  );
  state.pendingFiles = pending;
  elements.composerInput.value = text;
  resizeComposer();
  renderAttachmentTray();
  renderMessages({ forceBottom: true });
}

function addOptimisticTurn(text, pending) {
  const now = new Date().toISOString();
  const attachments = pending.map((item, index) => ({
    id: `pending-${index}`,
    original_name: item.file.name || "貼り付け画像",
    content_url: item.url,
  }));
  const localId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const userMessage = {
    id: `pending-user-${localId}`,
    role: "user",
    content: text,
    created_at: now,
    attachments,
    pending: true,
  };
  const assistantMessage = {
    id: `pending-assistant-${localId}`,
    role: "assistant",
    content: "",
    created_at: now,
    attachments: [],
    pending: true,
  };
  state.messages.push(userMessage, assistantMessage);
  state.pendingFiles = [];
  elements.composerInput.value = "";
  resizeComposer();
  renderAttachmentTray();
  renderMessages({ forceBottom: true });
  return { userMessage, assistantMessage };
}

function appendStreamDelta(message, text) {
  message.content += text;
  if (state.renderFrame !== null) return;
  state.renderFrame = requestAnimationFrame(() => {
    state.renderFrame = null;
    const body = elements.messages.querySelector(`[data-message-id="${CSS.escape(message.id)}"] .message-body`);
    if (body) renderAssistant(body, message.content, true);
    if (state.nearBottom) scrollToBottom(false);
    else elements.jumpLatest.hidden = false;
  });
}

async function refreshAfterTurn(conversationId) {
  state.activeId = conversationId;
  const [detail, current] = await Promise.all([
    apiJson(`/api/conversations/${conversationId}`),
    apiJson("/api/state"),
  ]);
  revokePendingUrls();
  state.messages = detail.messages;
  state.conversations = current.conversations;
  const active = activeConversation();
  renderConversationList();
  if (active) renderActiveHeader(active);
  renderMessages({ forceBottom: state.nearBottom });
}

function stopGeneration() {
  if (!state.generating || !state.controller) return;
  state.controller.abort();
  showToast("停止を要求しました。保存状態を同期します。");
}

async function readNdjson(stream, onEvent) {
  const reader = stream.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let buffer = "";
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      let newline;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        if (line) await onEvent(JSON.parse(line));
      }
      if (done) break;
    }
    if (buffer.trim()) throw new Error("終端改行のないストリームを受信しました。");
  } finally {
    reader.releaseLock();
  }
}

function addFiles(files) {
  if (!state.config?.images_enabled) return;
  for (const file of files) {
    if (!file.type.startsWith("image/")) continue;
    if (file.size > state.config.max_file_size_mb * 1024 * 1024) {
      showToast(`${file.name || "画像"} はファイル容量上限を超えています。`, "error");
      continue;
    }
    if (state.pendingFiles.length >= state.config.max_images) {
      showToast(`画像は${state.config.max_images}枚まで添付できます。`, "error");
      break;
    }
    state.pendingFiles.push({ file, url: URL.createObjectURL(file) });
  }
  renderAttachmentTray();
  updateComposerState();
}

function renderAttachmentTray() {
  elements.attachmentTray.replaceChildren();
  elements.attachmentTray.hidden = state.pendingFiles.length === 0;
  state.pendingFiles.forEach((item, index) => {
    const preview = document.createElement("div");
    preview.className = "attachment-preview";
    const image = document.createElement("img");
    image.src = item.url;
    image.alt = item.file.name || `添付画像 ${index + 1}`;
    const name = document.createElement("span");
    name.className = "attachment-preview-name";
    name.textContent = item.file.name || "貼り付け画像";
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove-attachment";
    remove.setAttribute("aria-label", `${name.textContent}を削除`);
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      URL.revokeObjectURL(item.url);
      state.pendingFiles.splice(index, 1);
      renderAttachmentTray();
      updateComposerState();
    });
    preview.append(image, name, remove);
    elements.attachmentTray.append(preview);
  });
}

function revokePendingUrls() {
  const urls = new Set();
  for (const message of state.messages) {
    for (const attachment of message.attachments || []) {
      if (String(attachment.content_url || "").startsWith("blob:")) urls.add(attachment.content_url);
    }
  }
  for (const url of urls) URL.revokeObjectURL(url);
}

function handlePaste(event) {
  const files = [...(event.clipboardData?.items || [])]
    .filter((item) => item.kind === "file" && item.type.startsWith("image/"))
    .map((item) => item.getAsFile())
    .filter(Boolean);
  if (files.length) {
    event.preventDefault();
    addFiles(files);
  }
}

function handleDragEnter(event) {
  event.preventDefault();
  state.dragDepth += 1;
  elements.composerZone.classList.add("dragging");
}

function handleDragLeave(event) {
  event.preventDefault();
  state.dragDepth = Math.max(0, state.dragDepth - 1);
  if (state.dragDepth === 0) elements.composerZone.classList.remove("dragging");
}

function handleDrop(event) {
  event.preventDefault();
  state.dragDepth = 0;
  elements.composerZone.classList.remove("dragging");
  addFiles([...event.dataTransfer.files]);
}

function handleComposerKeydown(event) {
  if (event.key === "Enter" && event.shiftKey && !event.isComposing && !state.composing) {
    event.preventDefault();
    sendMessage();
  }
}

function updateComposerState() {
  const active = activeConversation();
  const reason = composerBlockReason();
  elements.composerInput.disabled = !state.config;
  elements.attachButton.disabled = !state.config || state.generating || !state.config?.images_enabled;
  elements.fileInput.disabled = elements.attachButton.disabled;
  elements.sendButton.disabled = Boolean(reason);
  elements.sendReason.textContent = reason && reason !== "本文または画像を入力してください。" ? reason : "";
}

function composerBlockReason() {
  if (!state.config) return "設定を読み込んでいます。";
  if (state.generating) return "回答を生成中です。";
  if (!state.config.llm_configured) return "APIキーが未設定です。";
  const conversation = activeConversation();
  if (conversation && !conversation.continuable) return "この会話は継続できません。新しいチャットを開始してください。";
  if (elements.composerInput.value.length > state.config.max_text_length) return "本文が文字数上限を超えています。";
  const requestBytes = state.pendingFiles.reduce((sum, item) => sum + item.file.size, 0) + new Blob([elements.composerInput.value]).size;
  if (requestBytes > state.config.max_request_size_mb * 1024 * 1024) return "リクエスト全体の容量が上限を超えています。";
  if (!elements.composerInput.value.trim() && !state.pendingFiles.length) return "本文または画像を入力してください。";
  return "";
}

function setGenerating(value) {
  state.generating = value;
  elements.streamStatus.textContent = value ? "回答を生成しています。" : "回答生成が終了しました。";
  elements.stopButton.hidden = !value;
  elements.sendButton.hidden = value;
  elements.newChat.disabled = value;
  const active = activeConversation();
  if (active) renderActiveHeader(active);
  else renderDraftHeader();
  updateComposerState();
}

function resizeComposer() {
  elements.composerInput.rows = 1;
  const style = window.getComputedStyle(elements.composerInput);
  const lineHeight = Number.parseFloat(style.lineHeight) || 22;
  const verticalPadding = Number.parseFloat(style.paddingTop) + Number.parseFloat(style.paddingBottom);
  const rows = Math.ceil(Math.max(0, elements.composerInput.scrollHeight - verticalPadding) / lineHeight);
  elements.composerInput.rows = Math.max(1, Math.min(8, rows));
}

function handleMessageScroll() {
  const distance = elements.messageStage.scrollHeight - elements.messageStage.scrollTop - elements.messageStage.clientHeight;
  state.nearBottom = distance < 90;
  if (state.nearBottom) elements.jumpLatest.hidden = true;
}

function scrollToBottom(smooth) {
  elements.messageStage.scrollTo({
    top: elements.messageStage.scrollHeight,
    behavior: smooth ? "smooth" : "auto",
  });
  state.nearBottom = true;
  elements.jumpLatest.hidden = true;
}

function safeDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "—" : formatter.format(date);
}

function isSafeLink(value) {
  if (value.startsWith("/") || value.startsWith("#")) return true;
  try {
    const url = new URL(value, window.location.origin);
    return ["http:", "https:", "mailto:"].includes(url.protocol);
  } catch (_) {
    return false;
  }
}

function openSidebar() {
  if (!mobileViewport.matches) {
    elements.shell.classList.remove("sidebar-collapsed");
    syncSidebarState();
    return;
  }
  elements.shell.classList.add("sidebar-open");
  elements.sidebarScrim.hidden = false;
  syncSidebarState();
  elements.sidebarToggle.focus();
}

function closeSidebar(restoreFocus = false) {
  if (!mobileViewport.matches) return;
  elements.shell.classList.remove("sidebar-open");
  elements.sidebarScrim.hidden = true;
  syncSidebarState();
  if (restoreFocus) elements.openSidebar.focus();
}

function toggleSidebar() {
  if (mobileViewport.matches) {
    if (elements.shell.classList.contains("sidebar-open")) closeSidebar(true);
    else openSidebar();
    return;
  }
  elements.shell.classList.toggle("sidebar-collapsed");
  syncSidebarState();
}

function syncSidebarState() {
  const expanded = mobileViewport.matches
    ? elements.shell.classList.contains("sidebar-open")
    : !elements.shell.classList.contains("sidebar-collapsed");
  elements.sidebarToggle.setAttribute("aria-expanded", String(expanded));
  elements.sidebarToggle.setAttribute(
    "aria-label",
    mobileViewport.matches
      ? "サイドバーを閉じる"
      : expanded ? "サイドバーを折りたたむ" : "サイドバーを展開する",
  );
  elements.sidebar.inert = mobileViewport.matches && !expanded;
}

function handleSidebarViewportChange() {
  elements.shell.classList.remove("sidebar-open");
  elements.sidebarScrim.hidden = true;
  syncSidebarState();
}

function showBanner(message) {
  elements.systemBannerText.textContent = message;
  elements.systemBanner.hidden = false;
}

function showToast(message, kind = "info", duration = 4200) {
  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.textContent = message;
  elements.toastRegion.append(toast);
  setTimeout(() => toast.remove(), duration);
}

function reportError(error) {
  showToast(error?.message || "処理に失敗しました。", "error", 6500);
}

function askDeleteConfirm(anchor) {
  const popover = elements.deleteConfirm;
  popover.hidden = false;

  const position = () => {
    if (mobileViewport.matches) {
      popover.style.removeProperty("left");
      popover.style.removeProperty("top");
      return;
    }
    const anchorBounds = anchor.getBoundingClientRect();
    const popoverBounds = popover.getBoundingClientRect();
    const left = Math.min(
      window.innerWidth - popoverBounds.width - 10,
      Math.max(10, anchorBounds.right - popoverBounds.width),
    );
    const below = anchorBounds.bottom + 8;
    const top = below + popoverBounds.height <= window.innerHeight - 10
      ? below
      : Math.max(10, anchorBounds.top - popoverBounds.height - 8);
    popover.style.left = `${left}px`;
    popover.style.top = `${top}px`;
  };
  position();
  elements.deleteConfirmCancel.focus();

  return new Promise((resolve) => {
    const finish = (accepted) => {
      popover.hidden = true;
      popover.style.removeProperty("left");
      popover.style.removeProperty("top");
      elements.deleteConfirmCancel.removeEventListener("click", cancel);
      elements.deleteConfirmSubmit.removeEventListener("click", confirm);
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", keydown);
      window.removeEventListener("resize", position);
      elements.conversationList.removeEventListener("scroll", position);
      if (anchor.isConnected) anchor.focus();
      resolve(accepted);
    };
    const cancel = () => finish(false);
    const confirm = () => finish(true);
    const outside = (event) => {
      if (!popover.contains(event.target) && event.target !== anchor) cancel();
    };
    const keydown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        cancel();
      } else if (event.key === "Enter") {
        event.preventDefault();
        confirm();
      }
    };
    elements.deleteConfirmCancel.addEventListener("click", cancel);
    elements.deleteConfirmSubmit.addEventListener("click", confirm);
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", keydown);
    window.addEventListener("resize", position);
    elements.conversationList.addEventListener("scroll", position, { passive: true });
  });
}

async function apiJson(path, options = {}) {
  const fetchOptions = { method: options.method || "GET", headers: { ...MUTATION_HEADERS } };
  if (Object.hasOwn(options, "json")) {
    fetchOptions.headers["Content-Type"] = "application/json";
    fetchOptions.body = JSON.stringify(options.json);
  }
  const response = await fetch(path, fetchOptions);
  if (!response.ok) throw await errorFromResponse(response);
  if (response.status === 204) return null;
  return response.json();
}

async function errorFromResponse(response) {
  let payload = null;
  try { payload = await response.json(); } catch (_) { /* Safe fallback below. */ }
  const error = new Error(payload?.error?.message || `処理に失敗しました（HTTP ${response.status}）。`);
  error.code = payload?.error?.code || "http_error";
  error.retryable = Boolean(payload?.error?.retryable);
  error.status = response.status;
  return error;
}
