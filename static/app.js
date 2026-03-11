// ===== PDF RAG DeepSeek OCR Chatbot - Frontend App =====

(function () {
    'use strict';

    // ===== State =====
    let currentConvId = null;
    let isProcessing = false;
    let selectedUploadFiles = [];
    let currentAbortController = null;

    // ===== DOM Elements =====
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const sidebar = $('#sidebar');
    const btnToggleSidebar = $('#btn-toggle-sidebar');
    const btnNewChat = $('#btn-new-chat');
    const convList = $('#conversation-list');
    const uploadZone = $('#upload-zone');
    const fileInput = $('#file-input');
    const ocrToggle = $('#ocr-toggle');
    const btnUpload = $('#btn-upload');
    const uploadStatus = $('#upload-status');
    const fileListEl = $('#file-list');
    const customFileSelector = $('#custom-file-selector');
    const fileSelectorBtn = $('#file-selector-btn');
    const fileSelectorText = $('#file-selector-text');
    const customFileList = $('#custom-file-list');
    const previewModal = $('#preview-modal');
    const btnClosePreview = $('#btn-close-preview');
    const previewBody = $('#preview-body');
    const previewTitle = $('#preview-title');
    const agenticToggle = $('#agentic-toggle');
    const chatMessages = $('#chat-messages');
    const welcomeScreen = $('#welcome-screen');
    const reasoningPanel = $('#reasoning-panel');
    const reasoningToggle = $('#reasoning-toggle');
    const reasoningContent = $('#reasoning-content');
    const msgInput = $('#msg-input');
    const btnSend = $('#btn-send');
    const btnStop = $('#btn-stop');
    const headerTitle = $('#header-title');

    // ===== Init =====
    async function init() {
        // Configure Marked.js
        if (window.marked && window.hljs) {
            marked.setOptions({
                highlight: function(code, lang) {
                    const language = hljs.getLanguage(lang) ? lang : 'plaintext';
                    return hljs.highlight(code, { language }).value;
                },
                breaks: true,
                gfm: true
            });
        }

        bindEvents();
        await loadConversations();
        await loadFiles();
    }

    function bindEvents() {
        // Sidebar toggle
        btnToggleSidebar.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
        });

        // New chat
        btnNewChat.addEventListener('click', newConversation);

        // Upload zone
        uploadZone.addEventListener('click', () => fileInput.click());
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });
        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            handleFileSelect(e.dataTransfer.files);
        });
        fileInput.addEventListener('change', () => handleFileSelect(fileInput.files));
        btnUpload.addEventListener('click', uploadFiles);

        msgInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        msgInput.addEventListener('input', autoResize);
        btnSend.addEventListener('click', sendMessage);
        
        // Stop generation
        btnStop.addEventListener('click', () => {
            if (currentAbortController) {
                currentAbortController.abort();
                stopGenerationUI();
            }
        });

        // Reasoning toggle
        reasoningToggle.addEventListener('click', () => {
            reasoningPanel.classList.toggle('expanded');
        });

        // customFileSelector dropdown
        fileSelectorBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            customFileSelector.classList.toggle('open');
        });

        // Close dropdown outside click
        document.addEventListener('click', (e) => {
            if (!customFileSelector.contains(e.target)) {
                customFileSelector.classList.remove('open');
            }
        });

        // Preview modal close
        btnClosePreview.addEventListener('click', closePreview);
        previewModal.addEventListener('click', (e) => {
            if (e.target === previewModal) closePreview();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && previewModal.classList.contains('show')) closePreview();
        });
    }

    function closePreview() {
        previewModal.classList.remove('show');
        setTimeout(() => {
            previewModal.style.display = 'none';
            previewBody.innerHTML = '';
        }, 300); // clear after animation
    }

    function openPreview(filename) {
        previewTitle.textContent = filename;
        previewBody.innerHTML = '';
        
        const ext = filename.split('.').pop().toLowerCase();
        const url = '/uploads/' + encodeURIComponent(filename);
        
        if (['png', 'jpg', 'jpeg'].includes(ext)) {
            previewBody.innerHTML = `<img src="${url}" alt="Preview" onerror="this.outerHTML='<div class=\\'preview-error\\'>Preview not available. File might have been deleted.</div>'">`;
        } else if (ext === 'pdf') {
            previewBody.innerHTML = `<iframe src="${url}" onerror="this.outerHTML='<div class=\\'preview-error\\'>Preview not available. File might have been deleted.</div>'"></iframe>`;
        } else {
            previewBody.innerHTML = `<div class="preview-error">Format not supported for preview.</div>`;
        }
        
        // Show modal
        previewModal.style.display = 'flex';
        // Trigger reflow
        previewModal.offsetWidth;
        previewModal.classList.add('show');
    }

    // ===== Conversations =====
    async function loadConversations() {
        try {
            const res = await fetch('/api/conversations');
            const data = await res.json();
            renderConversations(data.conversations);

            // Auto-select latest or none
            if (data.conversations.length > 0 && !currentConvId) {
                await selectConversation(data.conversations[0].id);
            }
        } catch (e) {
            console.error('Load conversations error:', e);
        }
    }

    function renderConversations(convs) {
        convList.innerHTML = '';
        if (convs.length === 0) {
            convList.innerHTML = '<span class="empty-hint">No chat history</span>';
            return;
        }

        convs.forEach(c => {
            const item = document.createElement('div');
            item.className = 'conv-item' + (c.id === currentConvId ? ' active' : '');
            item.innerHTML = `
                <span class="conv-title">${escapeHtml(c.title || 'New Chat')}</span>
                <button class="conv-delete" title="Delete">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="3 6 5 6 21 6"></polyline>
                        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>
                    </svg>
                </button>
            `;

            item.querySelector('.conv-title').addEventListener('click', () => {
                selectConversation(c.id);
            });

            item.querySelector('.conv-delete').addEventListener('click', (e) => {
                e.stopPropagation();
                deleteConversation(c.id);
            });

            convList.appendChild(item);
        });
    }

    async function newConversation() {
        try {
            const res = await fetch('/api/conversations', { method: 'POST' });
            const data = await res.json();
            currentConvId = data.id;
            headerTitle.textContent = 'New Chat';
            chatMessages.innerHTML = '';
            chatMessages.appendChild(createWelcomeScreen());
            reasoningPanel.style.display = 'none';
            reasoningContent.innerHTML = '';
            await loadConversations();
        } catch (e) {
            console.error('New conversation error:', e);
        }
    }

    async function selectConversation(convId) {
        try {
            currentConvId = convId;
            const res = await fetch(`/api/conversations/${convId}`);
            const data = await res.json();

            headerTitle.textContent = data.title || 'New Chat';
            renderMessages(data.messages || []);

            // Update active state in sidebar
            $$('.conv-item').forEach(el => el.classList.remove('active'));
            const items = $$('.conv-item');
            items.forEach(el => {
                if (el.querySelector('.conv-title').textContent === (data.title || 'New Chat')) {
                    el.classList.add('active');
                }
            });

            // Re-render sidebar to update active state properly
            await loadConversations();

        } catch (e) {
            console.error('Select conversation error:', e);
        }
    }

    async function deleteConversation(convId) {
        try {
            await fetch(`/api/conversations/${convId}`, { method: 'DELETE' });
            if (currentConvId === convId) {
                currentConvId = null;
                chatMessages.innerHTML = '';
                chatMessages.appendChild(createWelcomeScreen());
                headerTitle.textContent = 'New Chat';
            }
            await loadConversations();
        } catch (e) {
            console.error('Delete conversation error:', e);
        }
    }

    // ===== Messages =====
    function renderMessages(messages) {
        chatMessages.innerHTML = '';
        if (messages.length === 0) {
            chatMessages.appendChild(createWelcomeScreen());
            return;
        }

        if (welcomeScreen) welcomeScreen.remove();

        messages.forEach(msg => {
            appendMessage(msg.role, msg.content, false);
        });
        scrollToBottom();
    }

    function appendMessage(role, content, animate = true) {
        // Remove welcome screen if present
        const welcome = chatMessages.querySelector('.welcome-screen');
        if (welcome) welcome.remove();

        const msgEl = document.createElement('div');
        msgEl.className = `message ${role}`;
        if (!animate) msgEl.style.animation = 'none';

        const avatarText = role === 'user' ? 'U' : 'AI';
        
        // Build the inner HTML
        let innerHtml = `
            <div class="message-avatar">${avatarText}</div>
            <div class="message-content">${formatContent(content)}</div>
        `;
        
        // Add Copy button for AI messages
        if (role === 'assistant') {
            innerHtml += `
                <button class="btn-copy-msg" title="Copy message">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" class="copy-icon" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" class="check-icon" stroke="currentColor" stroke-width="2" style="display:none; color: var(--green);"><polyline points="20 6 9 17 4 12"></polyline></svg>
                </button>
            `;
        }

        msgEl.innerHTML = innerHtml;
        chatMessages.appendChild(msgEl);
        
        // Attach event listener for the copy button
        if (role === 'assistant') {
            const btnCopy = msgEl.querySelector('.btn-copy-msg');
            btnCopy.addEventListener('click', () => {
                navigator.clipboard.writeText(content).then(() => {
                    const copyIcon = btnCopy.querySelector('.copy-icon');
                    const checkIcon = btnCopy.querySelector('.check-icon');
                    copyIcon.style.display = 'none';
                    checkIcon.style.display = 'block';
                    setTimeout(() => {
                        copyIcon.style.display = 'block';
                        checkIcon.style.display = 'none';
                    }, 2000);
                });
            });
        }
        
        if (animate) scrollToBottom();
    }

    function appendTypingIndicator() {
        const typing = document.createElement('div');
        typing.className = 'message assistant';
        typing.id = 'typing-indicator';
        typing.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        `;
        chatMessages.appendChild(typing);
        scrollToBottom();
    }

    function removeTypingIndicator() {
        const indicator = $('#typing-indicator');
        if (indicator) indicator.remove();
    }

    function createWelcomeScreen() {
        const el = document.createElement('div');
        el.className = 'welcome-screen';
        el.id = 'welcome-screen';
        el.innerHTML = `
            <div class="welcome-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            </div>
            <h2>PDF RAG DeepSeek OCR</h2>
            <p>Upload PDF documents or images, then ask questions.</p>
            <div class="welcome-tips">
                <div class="tip"><strong>Upload</strong><span>Drag & drop files to sidebar</span></div>
                <div class="tip"><strong>Chat</strong><span>Ask questions about documents</span></div>
                <div class="tip"><strong>OCR</strong><span>Enable OCR for images/scans</span></div>
            </div>
        `;
        return el;
    }

    // ===== Chat =====
    async function sendMessage() {
        const message = msgInput.value.trim();
        if (!message || isProcessing) return;

        isProcessing = true;
        btnSend.style.display = 'none';
        btnStop.style.display = 'flex';
        msgInput.value = '';
        msgInput.style.height = 'auto';

        // Show user message
        appendMessage('user', message);
        appendTypingIndicator();

        // Get selected files from filter
        const selectedFiles = Array.from($$('.file-checkbox:checked')).map(cb => cb.value);
        
        currentAbortController = new AbortController();

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: message,
                    conversation_id: currentConvId,
                    selected_files: selectedFiles.length > 0 ? selectedFiles : null,
                    use_agentic: agenticToggle.checked
                }),
                signal: currentAbortController.signal
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Parse SSE data
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            handleChatResponse(data);
                        } catch (e) {
                            // partial JSON, continue
                        }
                    }
                }
            }

            // Process remaining buffer
            if (buffer.startsWith('data: ')) {
                try {
                    const data = JSON.parse(buffer.slice(6));
                    handleChatResponse(data);
                } catch (e) {
                    // ignore
                }
            }

        } catch (e) {
            if (e.name === 'AbortError') {
                console.log('Stream aborted natively');
            } else {
                console.error('Chat error:', e);
                removeTypingIndicator();
                appendMessage('assistant', `Connection error: ${e.message}`);
            }
        }

        stopGenerationUI();
    }
    
    function stopGenerationUI() {
        isProcessing = false;
        btnSend.style.display = 'flex';
        btnStop.style.display = 'none';
        currentAbortController = null;
        removeTypingIndicator();
        msgInput.focus();
    }

    function handleChatResponse(data) {
        removeTypingIndicator();

        // Update conversation ID
        if (data.conversation_id) {
            currentConvId = data.conversation_id;
        }

        // Show response
        if (data.response) {
            appendMessage('assistant', data.response);
        }

        // Update title
        if (data.title) {
            headerTitle.textContent = data.title;
            loadConversations(); // Refresh sidebar
        }

        // Show reasoning steps
        if (data.reasoning_steps && data.reasoning_steps.length > 0) {
            reasoningPanel.style.display = '';
            reasoningContent.innerHTML = '';
            data.reasoning_steps.forEach((step, i) => {
                const stepEl = document.createElement('div');
                stepEl.className = 'reasoning-step';
                stepEl.textContent = `${i + 1}. ${step}`;
                reasoningContent.appendChild(stepEl);
            });
        }
    }

    // ===== File Upload =====
    function handleFileSelect(files) {
        selectedUploadFiles = Array.from(files);
        btnUpload.disabled = selectedUploadFiles.length === 0;

        // Show selected file names
        const existing = uploadZone.querySelector('.selected-files');
        if (existing) existing.remove();

        if (selectedUploadFiles.length > 0) {
            const preview = document.createElement('div');
            preview.className = 'selected-files';
            selectedUploadFiles.forEach(f => {
                preview.innerHTML += `<span class="file-tag">${escapeHtml(f.name)}</span>`;
            });
            uploadZone.appendChild(preview);
        }
    }

    async function uploadFiles() {
        if (selectedUploadFiles.length === 0) return;

        btnUpload.disabled = true;
        btnUpload.innerHTML = '<span class="spinner"></span> Processing...';
        uploadStatus.innerHTML = '';

        const formData = new FormData();
        selectedUploadFiles.forEach(f => formData.append('files', f));
        formData.append('enable_ocr', ocrToggle.checked);

        try {
            const res = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            // Show results
            let html = '';
            data.results.forEach(r => {
                if (r.status === 'ok') {
                    html += `<div class="status-ok">✓ ${escapeHtml(r.name)} — ${r.chunks} chunks [${r.time}]</div>`;
                } else if (r.status === 'skipped') {
                    html += `<div class="status-warning">⚠ ${escapeHtml(r.name)} — ${r.message}</div>`;
                } else if (r.status === 'needs_ocr') {
                    html += `<div class="status-warning">⚠ ${escapeHtml(r.name)} — ${r.message}</div>`;
                } else {
                    html += `<div class="status-error">✗ ${escapeHtml(r.name)} — ${r.message || 'Error'}</div>`;
                }
            });
            uploadStatus.innerHTML = html;

            // Refresh file list
            await loadFiles();

            // Clean up
            selectedUploadFiles = [];
            fileInput.value = '';
            const preview = uploadZone.querySelector('.selected-files');
            if (preview) preview.remove();

        } catch (e) {
            uploadStatus.innerHTML = `<div class="status-error">Error: ${escapeHtml(e.message)}</div>`;
        }

        btnUpload.innerHTML = '<span>Process Files</span>';
        btnUpload.disabled = true;
    }

    async function loadFiles() {
        try {
            const res = await fetch('/api/files');
            const data = await res.json();

            // File list in sidebar
            fileListEl.innerHTML = '';
            if (data.files.length === 0) {
                fileListEl.innerHTML = '<span class="empty-hint">No files yet</span>';
            } else {
                data.files.forEach(fileObj => {
                    const f = typeof fileObj === 'object' ? fileObj.name : fileObj;
                    const item = document.createElement('div');
                    item.className = 'file-item';
                    item.textContent = f;
                    fileListEl.appendChild(item);
                });
            }

            // Update custom file selector
            customFileList.innerHTML = '';
            if (data.files.length === 0) {
                customFileList.innerHTML = '<span class="empty-hint">No files</span>';
                fileSelectorText.textContent = 'No files';
            } else {
                data.files.forEach((fileObj, idx) => {
                    const fname = typeof fileObj === 'object' ? fileObj.name : fileObj;
                    const hasPreview = typeof fileObj === 'object' ? (fileObj.hasPreview !== false) : true;

                    const item = document.createElement('div');
                    item.className = 'custom-file-item';
                    
                    let previewBtn = '';
                    if (hasPreview) {
                        previewBtn = `
                        <button class="btn-preview" data-file="${escapeHtml(fname)}" title="Preview">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                        </button>`;
                    } else {
                        previewBtn = `
                        <button class="btn-preview disabled" style="opacity: 0.3; cursor: not-allowed;" title="Preview not available (original file was deleted by old system)">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>
                        </button>`;
                    }

                    let deleteBtn = `
                        <button class="btn-delete-file" data-file="${escapeHtml(fname)}" title="Delete file">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"></path></svg>
                        </button>
                    `;

                    item.innerHTML = `
                        <label>
                            <input type="checkbox" value="${escapeHtml(fname)}" class="file-checkbox">
                            <span class="file-name" title="${escapeHtml(fname)}">${escapeHtml(fname)}</span>
                        </label>
                        <div style="display: flex; gap: 4px;">
                            ${previewBtn}
                            ${deleteBtn}
                        </div>
                    `;
                    customFileList.appendChild(item);
                });
                
                // Add event listeners for checkboxes and previews
                $$('.file-checkbox').forEach(cb => {
                    cb.addEventListener('change', updateFileSelectorText);
                });
                $$('.btn-preview:not(.disabled)').forEach(btn => {
                    btn.addEventListener('click', (e) => {
                        e.stopPropagation();
                        openPreview(btn.dataset.file);
                    });
                });
                $$('.btn-delete-file').forEach(btn => {
                    btn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        if (confirm(`Are you sure you want to delete ${btn.dataset.file}?`)) {
                            const originalHtml = btn.innerHTML;
                            btn.innerHTML = '<span class="spinner" style="width: 14px; height: 14px; border-width: 2px;"></span>';
                            btn.disabled = true;
                            try {
                                const response = await fetch(`/api/files/${encodeURIComponent(btn.dataset.file)}`, { method: 'DELETE' });
                                if (response.ok) {
                                    loadFiles(); // Refresh UI
                                } else {
                                    alert('Error deleting file');
                                    btn.innerHTML = originalHtml;
                                    btn.disabled = false;
                                }
                            } catch (err) {
                                alert('Error deleting file: ' + err.message);
                                btn.innerHTML = originalHtml;
                                btn.disabled = false;
                            }
                        }
                    });
                });
                updateFileSelectorText();
            }

        } catch (e) {
            console.error('Load files error:', e);
        }
    }

    function updateFileSelectorText() {
        const checked = $$('.file-checkbox:checked');
        if (checked.length === 0) {
            fileSelectorText.textContent = 'All Files';
        } else if (checked.length === 1) {
            fileSelectorText.textContent = checked[0].value;
        } else {
            fileSelectorText.textContent = `${checked.length} files selected`;
        }
    }

    // ===== Helpers =====
    function formatContent(text) {
        if (!text) return '';
        
        if (window.marked && window.DOMPurify) {
            try {
                // Parse markdown
                const rawHtml = marked.parse(text);
                // Sanitize HTML
                return DOMPurify.sanitize(rawHtml, {
                    ADD_ATTR: ['target'] // Allow attributes if necessary
                });
            } catch (e) {
                console.error("Markdown parsing error:", e);
                return escapeHtml(text);
            }
        } else {
            // Fallback to basic if libraries are missing
            console.warn("Marked or DOMPurify not loaded. Using basic formatting.");
            let html = escapeHtml(text);
            html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
            html = html.replace(/\n/g, '<br>');
            html = html.replace(/---/g, '<hr style="border:none;border-top:1px solid var(--border-light);margin:8px 0;">');
            html = html.replace(/• /g, '<span style="color:var(--accent-light)">•</span> ');
            return html;
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        });
    }

    function autoResize() {
        msgInput.style.height = 'auto';
        msgInput.style.height = Math.min(msgInput.scrollHeight, 150) + 'px';
    }

    // ===== Boot =====
    document.addEventListener('DOMContentLoaded', init);
})();
