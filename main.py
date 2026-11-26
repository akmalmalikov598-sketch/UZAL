
import os
import time
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="UZAI",
    description="O'zbek tiliga ixtisoslashgan sun'iy intellekt",
    version="1.0.0"
)

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_MODEL = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen2.5-7B-Instruct:fastest"
)

API_URL = "https://router.huggingface.co/v1/chat/completions"

SYSTEM_PROMPT = """
Sening isming UZAI.

Sen O'zbekistonda yaratilayotgan, o'zbek tiliga
ixtisoslashgan sun'iy intellekt yordamchisisan.

ASOSIY QOIDALAR:

1. Har doim birinchi navbatda o'zbek tilida javob ber.
2. O'zbek tilining lotin va kirill yozuvlarini tushun.
3. O'zbek tili grammatikasi, imlosi va tinish
   belgilariga rioya qil.
4. Javoblaring tabiiy, ravon va tushunarli bo'lsin.
5. Foydalanuvchi ruscha yoki inglizcha yozsa ham,
   o'zbekcha javob ber, agar boshqa tilni so'ramasa.
6. O'zbek xalqining tarixi, madaniyati, urf-odatlari,
   geografiyasi va adabiyotiga hurmat bilan yondash.
7. Bilmagan narsangni to'qib chiqarma.
8. Aniq bilmasang, buni ochiq ayt.
9. Savollarga imkon qadar batafsil va foydali javob ber.
10. Dasturlash kodlarini to'g'ri va ishlaydigan shaklda yoz.
11. O'zbekcha sheva yoki norasmiy so'zlarni ham
    imkon qadar tushunishga harakat qil.
12. Foydalanuvchiga hurmat bilan murojaat qil.

SENING MAQSADING:
O'zbek tilida eng foydali va sifatli AI yordamchiga
aylanish.
"""

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    history: list[dict] = Field(default_factory=list)

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML

@app.get("/health")
async def health():
    return {
        "status": "online",
        "name": "UZAI",
        "version": "1.0.0",
        "model": HF_MODEL
    }

@app.post("/chat")
async def chat(data: ChatRequest):

    if not HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="AI API kaliti sozlanmagan. Render Environment bo'limiga HF_TOKEN kiriting."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Faqat oxirgi 12 ta xabarni kontekstda saqlaymiz
    for item in data.history[-12:]:
        role = item.get("role")
        content = item.get("content", "")

        if role in ("user", "assistant") and isinstance(content, str):
            messages.append({
                "role": role,
                "content": content[:5000]
            })

    messages.append({
        "role": "user",
        "content": data.message
    })

    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 1200,
        "top_p": 0.9
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                API_URL,
                headers=headers,
                json=payload
            )

        if response.status_code != 200:
            print("Hugging Face error:", response.text[:1000])
            raise HTTPException(
                status_code=502,
                detail="AI model hozir javob bera olmadi. Keyinroq urinib ko'ring."
            )

        result = response.json()
        answer = result["choices"][0]["message"]["content"]

        return {
            "answer": answer,
            "model": HF_MODEL,
            "name": "UZAI"
        }

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI javobi juda kechikdi. Qayta urinib ko'ring."
        )
    except HTTPException:
        raise
    except Exception as error:
        print("UZAI error:", str(error))
        raise HTTPException(
            status_code=500,
            detail="Kutilmagan xatolik yuz berdi."
        )


HTML = r"""
<!DOCTYPE html>
<html lang="uz">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#090d18">
<title>UZAI — O'zbek sun'iy intellekti</title>
<style>
*{box-sizing:border-box}
body{
 margin:0;background:#090d18;color:#f4f5fa;
 font-family:Inter,Arial,sans-serif;
 height:100vh;display:flex;flex-direction:column;
}
header{
 height:68px;border-bottom:1px solid #20283a;
 display:flex;align-items:center;justify-content:space-between;
 padding:0 22px;background:#0c1220;
}
.logo{font-size:25px;font-weight:900;letter-spacing:2px;color:#70d8c5}
.status{font-size:12px;color:#7ee2b8}
main{
 flex:1;overflow-y:auto;width:100%;max-width:900px;
 margin:auto;padding:25px 20px 130px;
}
.welcome{text-align:center;margin:80px auto 35px}
.welcome h1{font-size:42px;margin:0 0 12px}
.welcome h1 span{color:#70d8c5}
.welcome p{color:#98a4b9;font-size:15px}
.message{
 padding:16px 19px;border-radius:17px;margin:15px 0;
 max-width:85%;white-space:pre-wrap;overflow-wrap:anywhere;
 line-height:1.7;font-size:15px;
}
.user{
 margin-left:auto;background:#20334a;border:1px solid #334d69;
}
.bot{
 margin-right:auto;background:#131c2b;
 border:1px solid #263247;
}
.name{font-size:11px;color:#70d8c5;margin-bottom:8px;font-weight:bold}
.composer{
 position:fixed;bottom:0;left:0;right:0;
 padding:15px 20px 22px;
 background:linear-gradient(transparent,#090d18 25%);
}
.inputbox{
 max-width:850px;margin:auto;display:flex;gap:10px;
 background:#151e2d;border:1px solid #34445b;
 border-radius:17px;padding:10px;
}
textarea{
 flex:1;background:transparent;color:white;border:0;
 outline:none;resize:none;min-height:45px;max-height:130px;
 font:15px Arial;padding:12px;
}
button{
 background:#70d8c5;color:#07141a;border:0;
 border-radius:12px;padding:0 20px;font-weight:bold;
 cursor:pointer;font-size:14px;
}
button:disabled{opacity:.5}
.footer{text-align:center;color:#66758b;font-size:11px;margin-top:10px}
.actions{display:flex;gap:10px}
.small{
 background:transparent;color:#9baac0;
 border:1px solid #354157;padding:8px 12px;
}
@media(max-width:600px){
 .welcome{margin-top:50px}
 .welcome h1{font-size:34px}
 .message{max-width:95%}
 header{padding:0 15px}
}
</style>
</head>
<body>
<header>
 <div class="logo">UZAI</div>
 <div class="status">● AI ONLINE</div>
</header>

<main id="chat">
 <div class="welcome" id="welcome">
  <h1>Salom! Men <span>UZAI</span></h1>
  <p>O'zbek tilida fikrlash va suhbatlashish uchun yaratilgan AI.</p>
 </div>
</main>

<div class="composer">
 <div class="inputbox">
  <textarea id="input" placeholder="UZAI ga savolingizni yozing..."
  rows="1"></textarea>
  <button id="send" onclick="sendMessage()">Yuborish ↑</button>
 </div>
 <div class="footer">UZAI 1.0 · O'zbek tili uchun AI</div>
 <div class="actions" style="justify-content:center;margin-top:8px">
  <button class="small" onclick="clearChat()">Yangi suhbat</button>
 </div>
</div>

<script>
const chat = document.getElementById("chat");
const input = document.getElementById("input");
const send = document.getElementById("send");

let history = [];

function addMessage(role, text){
 document.getElementById("welcome")?.remove();

 const box = document.createElement("div");
 box.className = "message " + role;

 const name = document.createElement("div");
 name.className = "name";
 name.textContent = role === "user" ? "SIZ" : "UZAI";

 const content = document.createElement("div");
 content.textContent = text;

 box.append(name, content);
 chat.appendChild(box);
 chat.scrollTop = chat.scrollHeight;
}

async function sendMessage(){
 const message = input.value.trim();
 if(!message || send.disabled) return;

 addMessage("user", message);
 input.value = "";
 send.disabled = true;
 send.textContent = "...";

 const loading = document.createElement("div");
 loading.className = "message bot";
 loading.textContent = "UZAI javob tayyorlamoqda...";
 chat.appendChild(loading);

 try{
  const response = await fetch("/chat",{
   method:"POST",
   headers:{"Content-Type":"application/json"},
   body:JSON.stringify({
    message:message,
    history:history
   })
  });

  const data = await response.json();
  loading.remove();

  if(!response.ok){
   throw new Error(data.detail || "Server xatosi");
  }

  const answer = data.answer;
  addMessage("bot",answer);

  history.push(
   {role:"user",content:message},
   {role:"assistant",content:answer}
  );

  history = history.slice(-12);

 }catch(error){
  loading.remove();
  addMessage("bot","Xatolik: "+error.message);
 }

 send.disabled = false;
 send.textContent = "Yuborish ↑";
 input.focus();
}

function clearChat(){
 history=[];
 chat.innerHTML="";
 const welcome=document.createElement("div");
 welcome.className="welcome";
 welcome.id="welcome";
 welcome.innerHTML='<h1>Salom! Men <span>UZAI</span></h1><p>Yangi suhbatni boshlaymiz!</p>';
 chat.appendChild(welcome);
}

input.addEventListener("keydown",e=>{
 if(e.key==="Enter" && !e.shiftKey){
  e.preventDefault();
  sendMessage();
 }
});
</script>
</body>
</html>
"""
