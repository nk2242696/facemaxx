# 🆓 Free AI Setup Guide for FaceMaxx

## Quick Start (5 minutes)

### 1. 🚀 **Groq** (Recommended - Fastest & Best Free Tier)

1. **Sign up**: Go to https://console.groq.com/
2. **Get API Key**: Navigate to "API Keys" → Create new key
3. **Add to project**: 
   ```bash
   # Edit .env.local file
   VITE_GROQ_API_KEY=gsk_your-actual-key-here
   ```
4. **Restart server**: `npm run dev`

**Why Groq?**
- ✅ Very fast inference (2-10x faster than others)
- ✅ High rate limits on free tier
- ✅ Great quality with Llama 3.1 models
- ✅ No credit card required

---

### 2. 🤗 **Hugging Face** (Alternative)

1. **Sign up**: Go to https://huggingface.co/
2. **Get Token**: Settings → Access Tokens → Create new token
3. **Add to project**:
   ```bash
   VITE_HF_API_KEY=hf_your-token-here
   ```

---

### 3. 🔵 **Cohere** (Alternative)

1. **Sign up**: Go to https://cohere.ai/
2. **Get API Key**: Dashboard → API Keys
3. **Add to project**:
   ```bash
   VITE_COHERE_API_KEY=your-cohere-key-here
   ```

---

## Current Setup Features

- **Multi-Provider Fallback**: Tries providers in order until one works
- **Enhanced Offline Mode**: Amazing recommendations even without AI
- **Smart Error Handling**: Users never see API failures
- **Rate Limit Protection**: Automatic fallback when limits hit

## No API Key? No Problem!

The app works perfectly without any AI API - it uses our enhanced recommendation engine that provides professional-grade advice based on facial analysis scores.

## Need Help?

- Check console (F12) for debugging info
- All providers failed? The enhanced recommendations are actually excellent!
- Rate limited? Try a different provider or wait for reset

---

**🔥 Your FaceMaxz app is now ready to generate amazing personalized beauty recommendations!**