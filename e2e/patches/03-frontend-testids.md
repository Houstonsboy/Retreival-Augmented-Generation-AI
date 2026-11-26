# Patch 3: frontend test hooks (frontend/app/components/ChatView.tsx)

Six attributes, no visual change. Line numbers are from the audit; match on the code if they've shifted.

```diff
@@ ~1143  InputArea: textarea
   <textarea ref={inputRef} value={value} onChange={(e) => onChange(e.target.value)}
     onKeyDown={onKeyDown} onInput={onInput} placeholder={placeholder}
+    data-testid="chat-input"

@@ ~1153  InputArea: send button (icon-only)
   <button onClick={onSubmit} disabled={!value.trim() || isLoading}
+    data-testid="chat-submit"
+    aria-label="Send message"

@@ ~1173  UserBubble root
-  <div className="flex flex-col items-end gap-1">
+  <div className="flex flex-col items-end gap-1" data-testid="user-message">

@@ ~1194  LoadingCard root
-  <div className="rounded-2xl p-4" style={{ background: SURFACE, border: `1px solid ${BORDER}` }}>
+  <div className="rounded-2xl p-4" data-testid="chat-loading"
+    style={{ background: SURFACE, border: `1px solid ${BORDER}` }}>

@@ ~1320  AssistantCard: failed branch root
   <div className="rounded-2xl p-4 space-y-3"
+    data-testid="chat-error"
     style={{ background: SURFACE, border: `1.5px solid #F0C0C0` }}>

@@ ~1347  AssistantCard: success branch root
-  <div className="space-y-3">
+  <div className="space-y-3" data-testid="assistant-message">
```

`aria-label="Send message"` also gives screen-reader users a name for the icon-only button.

The tests detect these automatically. Without them, the tests fall back to Enter, the text "Drafting grounds from authorities…",
"Legal Analysis" and "Request failed".

## Optional: sign-in/page.tsx friendlyError()

`auth/unauthorized-domain` currently falls through to "An error occurred. Please try again."
Mapping it makes misconfigured deploys obvious to you (and to the test):

```diff
+    case "auth/unauthorized-domain":
+      return "This domain isn't authorised for sign-in. Add it in Firebase → Authentication → Authorized domains.";
```

## Not needed

`repository/caselaws/page.tsx:84` is an `<a href>` download to an unprotected route, not a fetch,
so it doesn't need `apiFetch`.
