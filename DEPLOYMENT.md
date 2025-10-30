# 🚀 FaceMaxx Deployment Guide - Vercel

## Prerequisites
- GitHub account
- Vercel account (sign up at [vercel.com](https://vercel.com))
- Your API keys ready (from .env.local)

---

## 🎯 Method 1: Deploy via Vercel Dashboard (Easiest)

### Step 1: Push to GitHub
```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - FaceMaxx app"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```

### Step 2: Import to Vercel
1. Go to [vercel.com/new](https://vercel.com/new)
2. Click **"Import Project"**
3. Select your GitHub repository
4. Vercel will auto-detect it's a Vite project ✨

### Step 3: Configure Environment Variables
Before deploying, add your API keys:

1. In Vercel dashboard, go to **Settings** → **Environment Variables**
2. Add these variables (use your actual API keys from `.env.local`):

```
Name: VITE_GROQ_API_KEY
Value: [Your Groq API key here]

Name: VITE_HF_API_KEY
Value: [Your Hugging Face API key here]

Name: VITE_COHERE_API_KEY
Value: [Your Cohere API key here]
```

3. Click **Save**

### Step 4: Deploy!
1. Click **"Deploy"**
2. Wait 1-2 minutes ⏱️
3. Your app is live! 🎉

Your URL will be: `https://your-project-name.vercel.app`

---

## 🎯 Method 2: Deploy via Vercel CLI

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Login
```bash
vercel login
```

### Step 3: Deploy
```bash
# First deployment (preview)
vercel

# Production deployment
vercel --prod
```

### Step 4: Add Environment Variables
```bash
# Add each variable
vercel env add VITE_GROQ_API_KEY
vercel env add VITE_HF_API_KEY
vercel env add VITE_COHERE_API_KEY
```

Or add them in the Vercel dashboard as shown in Method 1.

---

## 🔄 Continuous Deployment

Once set up, every time you push to GitHub:
1. Vercel automatically detects the push
2. Builds your app
3. Deploys to production
4. Done! ✨

```bash
# Make changes
git add .
git commit -m "Update UI styling"
git push

# Vercel deploys automatically! 🚀
```

---

## 🛠️ Custom Domain (Optional)

1. Go to your project in Vercel
2. Click **Settings** → **Domains**
3. Add your custom domain (e.g., `facemaxx.com`)
4. Follow DNS configuration steps
5. Vercel handles SSL automatically!

---

## 📊 Environment Variables Reference

Your app uses these environment variables:

| Variable | Purpose | Required |
|----------|---------|----------|
| `VITE_GROQ_API_KEY` | Primary AI provider (fast) | ✅ Yes |
| `VITE_HF_API_KEY` | Fallback AI provider | ✅ Yes |
| `VITE_COHERE_API_KEY` | Secondary fallback | ✅ Yes |

⚠️ **IMPORTANT**: Never commit `.env.local` to GitHub! It's already in `.gitignore`.

---

## 🔍 Troubleshooting

### Issue: Build fails
**Solution**: Check your build logs in Vercel dashboard. Common issues:
- Missing dependencies: Run `npm install` locally first
- Environment variables not set

### Issue: API not working in production
**Solution**: 
1. Check environment variables are added in Vercel
2. Make sure they have `VITE_` prefix
3. Redeploy after adding env vars

### Issue: 404 on refresh
**Solution**: Already handled by `vercel.json` configuration ✅

---

## 📈 Performance Optimizations

Your `vercel.json` includes:
- ✅ Cache headers for static assets (1 year)
- ✅ SPA routing (no 404s on refresh)
- ✅ Optimized for Vite builds

---

## 🎉 You're Ready!

Your FaceMaxx app is production-ready with:
- ✅ Secure API key handling
- ✅ Automatic HTTPS
- ✅ Global CDN
- ✅ Auto-deployments on push
- ✅ Free hosting!

**Happy deploying!** 🚀

---

## 🔗 Useful Links

- [Vercel Documentation](https://vercel.com/docs)
- [Vite Documentation](https://vitejs.dev/)
- [Environment Variables Guide](https://vercel.com/docs/concepts/projects/environment-variables)
