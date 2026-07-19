"use strict";

const elements = {
  shell: document.querySelector("#app-shell"),
  sidebar: document.querySelector("#sidebar"),
  sidebarScrim: document.querySelector("#sidebar-scrim"),
  openSidebar: document.querySelector("#open-sidebar"),
  sidebarToggle: document.querySelector("#sidebar-toggle"),
  appTitle: document.querySelector("#app-title"),
  newChat: document.querySelector("#new-chat"),
  conversationList: document.querySelector("#conversation-list"),
  conversationCount: document.querySelector("#conversation-count"),
  retentionDays: document.querySelector("#retention-days"),
  modelSelect: document.querySelector("#model-select"),
  systemBanner: document.querySelector("#system-banner"),
  systemBannerText: document.querySelector("#system-banner-text"),
  messageStage: document.querySelector("#message-stage"),
  messages: document.querySelector("#messages"),
  emptyState: document.querySelector("#empty-state"),
  jumpLatest: document.querySelector("#jump-latest"),
  composerZone: document.querySelector("#composer-zone"),
  composerInput: document.querySelector("#composer-input"),
  fileInput: document.querySelector("#file-input"),
  attachButton: document.querySelector("#attach-button"),
  attachmentTray: document.querySelector("#attachment-tray"),
  characterCount: document.querySelector("#character-count"),
  uploadLimits: document.querySelector("#upload-limits"),
  sendReason: document.querySelector("#send-reason"),
  sendButton: document.querySelector("#send-button"),
  stopButton: document.querySelector("#stop-button"),
  toastRegion: document.querySelector("#toast-region"),
  streamStatus: document.querySelector("#stream-status"),
  deleteConfirm: document.querySelector("#delete-confirm"),
  deleteConfirmTitle: document.querySelector("#delete-confirm-title"),
  deleteConfirmCancel: document.querySelector("#delete-confirm-cancel"),
  deleteConfirmSubmit: document.querySelector("#delete-confirm-submit"),
  confirmDialog: document.querySelector("#confirm-dialog"),
  dialogTitle: document.querySelector("#dialog-title"),
  dialogMessage: document.querySelector("#dialog-message"),
  dialogCancel: document.querySelector("#dialog-cancel"),
  dialogConfirm: document.querySelector("#dialog-confirm"),
  renameDialog: document.querySelector("#rename-dialog"),
  renameForm: document.querySelector("#rename-form"),
  renameInput: document.querySelector("#rename-input"),
};

const state = {
  config: null,
  conversations: [],
  activeId: null,
  draftModelKey: null,
  messages: [],
  pendingFiles: [],
  generating: false,
  controller: null,
  nearBottom: true,
  composing: false,
  renderFrame: null,
  dragDepth: 0,
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
    const [config, conversations] = await Promise.all([
      apiJson("/api/config/public"),
      apiJson("/api/conversations"),
    ]);
    state.config = config;
    state.conversations = conversations;
    applyConfig();
    renderConversationList();
    beginNewChat(config.default_model_key);
  } catch (error) {
    showToast(error.message || "アプリを初期化できませんでした。", "error", 8000);
    showBanner("ローカルサーバーへ接続できません。ページを再読み込みしてください。");
  }
}

function bindEvents() {
  elements.newChat.addEventListener("click", () => {
    const selected = elements.modelSelect.value || state.config?.default_model_key;
    beginNewChat(selected);
  });
  elements.openSidebar.addEventListener("click", openSidebar);
  elements.sidebarToggle.addEventListener("click", toggleSidebar);
  elements.sidebarScrim.addEventListener("click", () => closeSidebar(true));
  elements.modelSelect.addEventListener("change", handleModelChange);
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
  elements.appTitle.textContent = state.config.app_title;
  elements.retentionDays.textContent = `${state.config.history_days}日`;
  elements.composerInput.maxLength = state.config.max_text_length;
  elements.uploadLimits.textContent = state.config.images_enabled
    ? ` · 画像1枚${state.config.max_file_size_mb}MB／最大${state.config.max_images}枚／全体${state.config.max_request_size_mb}MB`
    : " · 画像添付は無効";
  elements.fileInput.accept = state.config.models
    .filter((model) => model.supports_images)
    .length > 0 ? "image/jpeg,image/png,image/webp" : "";
  renderModelOptions();
  if (!state.config.llm_configured) {
    showBanner("OPENAI_API_KEY が未設定です。履歴の閲覧と整理はできますが、新しいメッセージは送信できません。");
  }
}

function renderModelOptions() {
  elements.modelSelect.replaceChildren();
  for (const model of state.config?.models || []) {
    const option = document.createElement("option");
    option.value = model.key;
    option.textContent = model.label;
    elements.modelSelect.append(option);
  }
}

function beginNewChat(modelKey) {
  if (state.generating) return;
  state.activeId = null;
  state.draftModelKey = modelKey || state.config.default_model_key;
  state.messages = [];
  renderConversationList();
  renderDraftHeader();
  renderMessages({ forceBottom: true });
  updateComposerState();
  closeSidebar();
  elements.composerInput.focus();
}

async function materializeDraftConversation() {
  const conversation = await apiJson("/api/conversations", {
    method: "POST",
    json: { model_key: state.draftModelKey || state.config.default_model_key },
  });
  state.activeId = conversation.id;
  state.draftModelKey = null;
  state.conversations.unshift(conversation);
  renderConversationList();
  renderActiveHeader(conversation);
  return conversation;
}

async function refreshConversations() {
  state.conversations = await apiJson("/api/conversations");
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
  state.draftModelKey = null;
  const [conversation, messages] = await Promise.all([
    apiJson(`/api/conversations/${conversationId}`),
    apiJson(`/api/conversations/${conversationId}/messages`),
  ]);
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
  if ([...elements.modelSelect.options].some((option) => option.value === conversation.model_key)) {
    elements.modelSelect.value = conversation.model_key;
  } else {
    const missing = document.createElement("option");
    missing.value = conversation.model_key;
    missing.textContent = `${conversation.model_key}（利用不可）`;
    missing.disabled = true;
    elements.modelSelect.prepend(missing);
    elements.modelSelect.value = conversation.model_key;
  }
  const busy = state.generating || conversation.is_generating;
  elements.modelSelect.disabled = busy || !state.config.models.length;
}

function renderDraftHeader() {
  elements.modelSelect.value = state.draftModelKey || state.config.default_model_key;
  elements.modelSelect.disabled = state.generating || !state.config.models.length;
}

function renderConversationList() {
  elements.conversationList.replaceChildren();
  elements.conversationCount.textContent = String(state.conversations.length);
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

    const actions = document.createElement("div");
    actions.className = "conversation-actions";
    const rename = miniAction("編集", "会話タイトルを変更", () => openRename(conversation.id));
    const remove = miniAction(
      "×",
      "会話を削除",
      (button) => deleteConversation(conversation.id, button),
      "delete",
    );
    actions.append(rename, remove);
    item.append(open, actions);
    elements.conversationList.append(item);
  }
}

function miniAction(text, label, handler, extraClass = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `conversation-mini-action ${extraClass}`.trim();
  button.textContent = text;
  button.setAttribute("aria-label", label);
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    handler(button);
  });
  return button;
}

async function handleModelChange() {
  const conversation = activeConversation();
  const requested = elements.modelSelect.value;
  if (!conversation) {
    state.draftModelKey = requested;
    updateComposerState();
    return;
  }
  if (requested === conversation.model_key) return;
  if (conversation.has_messages) {
    const accepted = await askConfirm({
      title: "モデルを変更しますか？",
      message: "モデルを変更すると、AI側の会話コンテキストはリセットされます。\n画面上の履歴は残りますが、新しいモデルには自動で引き継がれません。",
      confirmLabel: "モデルを変更",
    });
    if (!accepted) {
      elements.modelSelect.value = conversation.model_key;
      return;
    }
  }
  try {
    const updated = await apiJson(`/api/conversations/${conversation.id}`, {
      method: "PATCH",
      json: { model_key: requested },
    });
    replaceConversation(updated);
    renderConversationList();
    renderActiveHeader(updated);
    renderMessages({ forceBottom: false });
    showToast("モデルを変更し、AIコンテキストをリセットしました。");
  } catch (error) {
    elements.modelSelect.value = conversation.model_key;
    reportError(error);
  }
  updateComposerState();
}

async function openRename(conversationId) {
  if (!conversationId || state.generating) return;
  const conversation = state.conversations.find((item) => item.id === conversationId);
  if (!conversation) return;
  if (conversation.is_generating) {
    showToast("回答生成中の会話は変更できません。", "error");
    return;
  }
  const returnFocus = document.activeElement;
  elements.renameInput.value = conversation.title;
  elements.renameDialog.returnValue = "cancel";
  elements.renameDialog.showModal();
  elements.renameInput.select();
  const result = await waitForDialog(elements.renameDialog);
  if (returnFocus instanceof HTMLElement) returnFocus.focus();
  if (result !== "confirm") return;
  const title = elements.renameInput.value.trim();
  if (!title || title.length > 100) {
    showToast("タイトルは1文字以上100文字以下で入力してください。", "error");
    return;
  }
  try {
    const updated = await apiJson(`/api/conversations/${conversationId}`, {
      method: "PATCH",
      json: { title },
    });
    replaceConversation(updated);
    renderConversationList();
    if (conversationId === state.activeId) renderActiveHeader(updated);
  } catch (error) {
    reportError(error);
  }
}

async function deleteConversation(conversationId, anchor) {
  if (!conversationId || state.generating) return;
  const conversation = state.conversations.find((item) => item.id === conversationId);
  if (!conversation) return;
  if (conversation.is_generating) {
    showToast("回答生成中の会話は削除できません。", "error");
    return;
  }
  const accepted = await askDeleteConfirm(conversation, anchor);
  if (!accepted) return;
  try {
    await apiJson(`/api/conversations/${conversationId}`, { method: "DELETE" });
    state.conversations = state.conversations.filter((item) => item.id !== conversationId);
    if (state.activeId === conversationId) {
      state.activeId = null;
      state.messages = [];
      if (state.conversations.length) await selectConversation(state.conversations[0].id);
      else beginNewChat(state.config.default_model_key);
    } else {
      renderConversationList();
    }
    showToast("会話を削除しました。");
  } catch (error) {
    reportError(error);
  }
}

function replaceConversation(conversation) {
  const index = state.conversations.findIndex((item) => item.id === conversation.id);
  if (index >= 0) state.conversations[index] = conversation;
  else state.conversations.unshift(conversation);
}

function renderMessages({ forceBottom = false } = {}) {
  const keepAtBottom = forceBottom || state.nearBottom;
  elements.messages.replaceChildren();
  const conversation = activeConversation();
  const activeEpoch = conversation?.context_epoch;
  let previousEpoch = null;
  let previousMessage = null;
  for (const message of state.messages) {
    if (previousEpoch !== null && message.context_epoch !== previousEpoch) {
      const modelLabel = message.context_epoch === activeEpoch ? conversation?.model_label : null;
      elements.messages.append(contextBoundary(modelLabel, contextResetCause(previousMessage)));
    }
    previousEpoch = message.context_epoch;
    previousMessage = message;
    elements.messages.append(renderMessage(message));
  }
  if (previousEpoch !== null && activeEpoch !== undefined && activeEpoch !== previousEpoch) {
    elements.messages.append(contextBoundary(
      conversation?.model_label,
      contextResetCause(previousMessage),
    ));
  }
  elements.emptyState.hidden = state.messages.length > 0;
  if (keepAtBottom) requestAnimationFrame(() => scrollToBottom(false));
  else elements.jumpLatest.hidden = true;
}

function contextResetCause(previousMessage) {
  return previousMessage?.error_code === "context_reference_lost"
    ? "reference_lost"
    : "model_change";
}

function contextBoundary(modelLabel, cause) {
  const boundary = document.createElement("div");
  boundary.className = "context-boundary";
  const label = document.createElement("span");
  if (cause === "reference_lost") {
    label.textContent = "AI側の会話コンテキストを継続できなかったため、ここから新しい文脈です";
  } else {
    label.textContent = modelLabel
      ? `モデルを${modelLabel}に変更しました。ここから新しい文脈です`
      : "モデルを変更しました。ここから新しい文脈です";
  }
  boundary.append(label);
  return boundary;
}

function renderMessage(message) {
  const excluded = message.status !== "streaming" && !message.included_in_context && !message.pending_context;
  const article = document.createElement("article");
  article.className = `message ${message.role}${excluded ? " excluded" : ""}`;
  article.dataset.messageId = message.id;

  const content = document.createElement("div");
  content.className = "message-content";
  const body = document.createElement("div");
  body.className = "message-body";
  if (message.role === "assistant") renderAssistant(body, message.content, message.status === "streaming");
  else body.textContent = message.content;

  if (message.attachments?.length) body.append(renderMessageImages(message.attachments));
  content.append(body);

  const meta = document.createElement("div");
  meta.className = "message-meta";
  if (message.status !== "completed") {
    const status = document.createElement("span");
    status.className = "status-badge";
    status.textContent = statusLabel(message.status);
    meta.append(status);
  }
  if (excluded) {
    const badge = document.createElement("span");
    badge.className = "excluded-badge";
    badge.textContent = "AIの次回コンテキストには含まれません";
    meta.append(badge);
  }
  const time = document.createElement("time");
  time.className = "message-time";
  time.dateTime = message.created_at;
  time.textContent = safeDate(message.created_at);
  meta.append(time);
  content.append(meta);
  if (message.error_message) {
    const error = document.createElement("div");
    error.className = "message-error";
    error.textContent = message.error_message;
    content.append(error);
  }
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
  for (const item of pending) form.append("images", item.file, item.file.name || "pasted-image.png");

  const optimisticTurn = addOptimisticTurn(text, pending);
  setGenerating(true);
  state.controller = new AbortController();
  let started = false;
  let terminal = false;
  let createdForSend = false;
  const submittedModelKey = state.draftModelKey || state.config.default_model_key;
  try {
    if (!state.activeId) {
      await materializeDraftConversation();
      createdForSend = true;
    }
    const response = await fetch(`/api/conversations/${state.activeId}/messages`, {
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
      if (terminal) throw new Error("終端イベントの後にデータを受信しました。");
      if (event.type === "start") {
        if (started) throw new Error("開始イベントが重複しています。");
        started = true;
        acceptOptimisticTurn(event, optimisticTurn);
      } else if (event.type === "delta") {
        if (!started) throw new Error("開始前に差分を受信しました。");
        appendStreamDelta(event.assistant_message_id, event.text || "");
      } else if (event.type === "completed") {
        if (!started) throw new Error("開始前に完了しました。");
        terminal = true;
      } else if (event.type === "error") {
        if (!started) throw new Error("開始前にエラーイベントを受信しました。");
        terminal = true;
        showToast(event.message || "回答生成に失敗しました。", "error", 7000);
      } else if (event.type === "cancelled") {
        if (!started) throw new Error("開始前に停止イベントを受信しました。");
        terminal = true;
      } else {
        throw new Error("未知のストリームイベントを受信しました。");
      }
    });
    if (!started || !terminal) throw new Error("回答ストリームが途中で終了しました。");
  } catch (error) {
    if (!state.activeId) restoreUnsentDraft(optimisticTurn, text, pending);
    if (error.name !== "AbortError") reportError(error);
  } finally {
    state.controller = null;
    try {
      if (started || state.activeId) {
        await refreshAfterTurn();
      }
      if (createdForSend && !started && state.messages.length === 0) {
        await discardEmptyMaterializedConversation(submittedModelKey, text, pending);
      }
    } catch (error) {
      reportError(error);
    } finally {
      setGenerating(false);
    }
  }
}

async function discardEmptyMaterializedConversation(modelKey, text, pending) {
  const emptyConversationId = state.activeId;
  if (emptyConversationId) {
    await apiJson(`/api/conversations/${emptyConversationId}`, { method: "DELETE" });
    state.conversations = state.conversations.filter((item) => item.id !== emptyConversationId);
  }
  beginNewChat(modelKey);
  state.pendingFiles = pending;
  elements.composerInput.value = text;
  resizeComposer();
  renderAttachmentTray();
  updateComposerState();
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
  const contextEpoch = activeConversation()?.context_epoch || 1;
  const localId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const userMessage = {
    id: `pending-user-${localId}`,
    role: "user",
    content: text,
    status: "completed",
    context_epoch: contextEpoch,
    included_in_context: false,
    error_code: null,
    error_message: null,
    created_at: now,
    updated_at: now,
    attachments,
    pending_context: true,
  };
  const assistantMessage = {
    id: `pending-assistant-${localId}`,
    role: "assistant",
    content: "",
    status: "streaming",
    context_epoch: contextEpoch,
    included_in_context: false,
    error_code: null,
    error_message: null,
    created_at: now,
    updated_at: now,
    attachments: [],
  };
  state.messages.push(userMessage, assistantMessage);
  state.pendingFiles = [];
  elements.composerInput.value = "";
  resizeComposer();
  renderAttachmentTray();
  renderMessages({ forceBottom: true });
  return { userMessage, assistantMessage };
}

function acceptOptimisticTurn(event, optimisticTurn) {
  optimisticTurn.userMessage.id = event.user_message_id;
  optimisticTurn.userMessage.context_epoch = event.context_epoch;
  optimisticTurn.assistantMessage.id = event.assistant_message_id;
  optimisticTurn.assistantMessage.context_epoch = event.context_epoch;
  renderMessages({ forceBottom: true });
}

function appendStreamDelta(messageId, text) {
  const message = state.messages.find((item) => item.id === messageId);
  if (!message) throw new Error("ストリーム対象メッセージが見つかりません。");
  message.content += text;
  if (state.renderFrame !== null) return;
  state.renderFrame = requestAnimationFrame(() => {
    state.renderFrame = null;
    const body = elements.messages.querySelector(`[data-message-id="${CSS.escape(messageId)}"] .message-body`);
    if (body) renderAssistant(body, message.content, true);
    if (state.nearBottom) scrollToBottom(false);
    else elements.jumpLatest.hidden = false;
  });
}

async function refreshAfterTurn() {
  if (!state.activeId) return;
  let messages = [];
  let conversations = [];
  for (let attempt = 0; attempt < 50; attempt += 1) {
    [messages, conversations] = await Promise.all([
      apiJson(`/api/conversations/${state.activeId}/messages`),
      apiJson("/api/conversations"),
    ]);
    const current = conversations.find((conversation) => conversation.id === state.activeId);
    if (
      !messages.some((message) => message.status === "streaming") &&
      !current?.is_generating
    ) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  revokePendingUrls();
  state.messages = messages;
  state.conversations = conversations;
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
    if (!state.config.models.some((model) => model.supports_images)) continue;
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
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing && !state.composing) {
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
  const length = elements.composerInput.value.length;
  elements.characterCount.textContent = `${length.toLocaleString("ja-JP")} / ${Number(state.config?.max_text_length || 100000).toLocaleString("ja-JP")}`;
  elements.characterCount.classList.toggle("over", length > (state.config?.max_text_length || 100000));
}

function composerBlockReason() {
  if (!state.config) return "設定を読み込んでいます。";
  if (state.generating) return "回答を生成中です。";
  if (!state.config.llm_configured) return "APIキーが未設定です。";
  const conversation = activeConversation();
  if (conversation?.is_generating) return "この会話では回答を生成中です。";
  if (conversation && !conversation.model_available) return "利用できるモデルを選び直してください。";
  const modelKey = conversation?.model_key || state.draftModelKey || state.config.default_model_key;
  const model = state.config.models.find((item) => item.key === modelKey);
  if (!model) return "利用できるモデルを選び直してください。";
  if (state.pendingFiles.length && !model?.supports_images) return "このモデルは画像入力に対応していません。";
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

function statusLabel(status) {
  return ({ streaming: "生成中", failed: "失敗", cancelled: "停止" })[status] || status;
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

function askConfirm({ title, message, confirmLabel }) {
  const returnFocus = document.activeElement;
  elements.dialogTitle.textContent = title;
  elements.dialogMessage.textContent = message;
  elements.dialogConfirm.textContent = confirmLabel;
  elements.confirmDialog.returnValue = "cancel";
  elements.confirmDialog.showModal();
  elements.dialogCancel.focus();
  return waitForDialog(elements.confirmDialog).then((value) => {
    if (returnFocus instanceof HTMLElement) returnFocus.focus();
    return value === "confirm";
  });
}

function askDeleteConfirm(conversation, anchor) {
  const popover = elements.deleteConfirm;
  elements.deleteConfirmTitle.textContent = conversation.title;
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

function waitForDialog(dialog) {
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue), { once: true });
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
