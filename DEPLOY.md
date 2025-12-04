# Deploy LST-DocAI to the Internet

## Option 1: Render (Free Tier - Recommended)

1. **Sign up** at https://render.com (free account)

2. **Create a new Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repo (or use Render's Git)
   - Or use "Deploy manually" and upload your code

3. **Configure the service**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3
   - **Port**: Render will set this automatically

4. **Deploy** - Render will give you a public URL like `https://your-app.onrender.com`

## Option 2: Railway (Free Tier)

1. **Sign up** at https://railway.app
2. **New Project** → "Deploy from GitHub repo"
3. Railway auto-detects Python and runs your app
4. Get public URL automatically

## Option 3: Fly.io (Free Tier)

1. **Install Fly CLI**: `iwr https://fly.io/install.ps1 -useb | iex`
2. **Login**: `fly auth login`
3. **Launch**: `fly launch` (in your project folder)
4. **Deploy**: `fly deploy`

## Option 4: ngrok (Quick Testing)

1. Download from https://ngrok.com/download
2. Extract and run: `ngrok http 8000`
3. Share the public URL (temporary, free tier has time limits)

## Notes

- For production, consider adding authentication/rate limiting
- Free tiers may have cold starts (first request after inactivity is slow)
- Models will download on first request (may take time)

