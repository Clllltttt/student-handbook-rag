const form = document.querySelector("#questionForm");
const input = document.querySelector("#questionInput");
const askButton = document.querySelector("#askButton");
const emptyState = document.querySelector("#emptyState");
const answerCard = document.querySelector("#answerCard");
const loadingCard = document.querySelector("#loadingCard");
const errorCard = document.querySelector("#errorCard");
const sourcesPanel = document.querySelector("#sourcesPanel");
const sourceList = document.querySelector("#sourceList");
const sourcePlaceholder = document.querySelector("#sourcePlaceholder");
const sourceTemplate = document.querySelector("#sourceTemplate");
let lastAnswer = "";

function pageLabel(source) {
  return source.printed_page_start === source.printed_page_end
    ? `手册第 ${source.printed_page_start} 页`
    : `手册第 ${source.printed_page_start}–${source.printed_page_end} 页`;
}

function setBusy(isBusy) {
  askButton.disabled = isBusy;
  input.disabled = isBusy;
  loadingCard.hidden = !isBusy;
  if (isBusy) {
    emptyState.hidden = true;
    answerCard.hidden = true;
    errorCard.hidden = true;
  }
}

function renderSources(sources) {
  sourceList.replaceChildren();
  sourcePlaceholder.hidden = sources.length > 0;
  document.querySelector("#sourceCount").textContent = sources.length
    ? `${sources.length} 条有效引用`
    : "本次无引用";

  sources.forEach((source, index) => {
    const node = sourceTemplate.content.cloneNode(true);
    const card = node.querySelector(".source-card");
    node.querySelector(".source-index").textContent = String(index + 1).padStart(2, "0");
    node.querySelector(".source-page").textContent = pageLabel(source);
    node.querySelector(".source-title").textContent = source.title;
    node.querySelector(".source-content").textContent = source.content;
    const link = node.querySelector(".source-link");
    link.href = `/handbook#page=${source.pdf_page_start}`;
    card.addEventListener("dblclick", () => card.classList.toggle("expanded"));
    sourceList.appendChild(node);
  });
}

function renderAnswer(data) {
  document.querySelector("#questionEcho").textContent = data.question;
  const answerCopy = document.querySelector("#answerCopy");
  answerCopy.textContent = data.answer;
  answerCopy.classList.toggle("is-refusal", !data.answerable);
  document.querySelector("#answerTime").textContent = `${(data.elapsed_ms / 1000).toFixed(1)} 秒`;
  document.querySelector("#modelName").textContent = `${data.model} · ${data.api_calls} 次生成调用`;
  lastAnswer = data.answer;
  renderSources(data.sources || []);
  answerCard.hidden = false;
}

async function askQuestion(question) {
  setBusy(true);
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "请求失败");
    renderAnswer(data);
  } catch (error) {
    document.querySelector("#errorText").textContent = error.message;
    errorCard.hidden = false;
    document.querySelector("#sourceCount").textContent = "查询失败";
  } finally {
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (question) askQuestion(question);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 130)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.textContent;
    form.requestSubmit();
  });
});

document.querySelector("#copyAnswer").addEventListener("click", async (event) => {
  if (!lastAnswer) return;
  await navigator.clipboard.writeText(lastAnswer);
  event.currentTarget.textContent = "已复制";
  setTimeout(() => { event.currentTarget.textContent = "复制回答"; }, 1400);
});

fetch("/api/status")
  .then((response) => response.json())
  .catch(() => {
    document.querySelector("#status").classList.add("offline");
    document.querySelector("#status span:last-child").textContent = "本地服务异常";
  });
