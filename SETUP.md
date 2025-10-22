# Authentication Setup Guide

This Next.js application includes OAuth authentication with Google and GitHub using NextAuth.js.

## Features

- Google OAuth login
- GitHub OAuth login
- Protected dashboard route
- Session management
- Modern UI with Tailwind CSS

## Prerequisites

- Node.js 18+ installed
- Google Cloud Console account
- GitHub account

## Installation

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
```bash
cp .env.local.example .env.local
```

## OAuth Configuration

### Google OAuth Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Navigate to "APIs & Services" > "Credentials"
4. Click "Create Credentials" > "OAuth client ID"
5. Configure OAuth consent screen if prompted
6. Select "Web application" as application type
7. Add authorized redirect URI:
   - `http://localhost:3000/api/auth/callback/google`
8. Copy the Client ID and Client Secret
9. Add them to `.env.local`:
```env
GOOGLE_CLIENT_ID=your-client-id-here
GOOGLE_CLIENT_SECRET=your-client-secret-here
```

### GitHub OAuth Setup

1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click "New OAuth App"
3. Fill in the form:
   - Application name: Your App Name
   - Homepage URL: `http://localhost:3000`
   - Authorization callback URL: `http://localhost:3000/api/auth/callback/github`
4. Click "Register application"
5. Copy the Client ID
6. Click "Generate a new client secret" and copy it
7. Add them to `.env.local`:
```env
GITHUB_CLIENT_ID=your-client-id-here
GITHUB_CLIENT_SECRET=your-client-secret-here
```

### Auth Secret

Generate a secure secret for NextAuth:

```bash
openssl rand -base64 32
```

Add it to `.env.local`:
```env
AUTH_SECRET=your-generated-secret-here
```

## Running the Application

Start the development server:

```bash
npm run dev
```

Visit `http://localhost:3000` to see your application.

## Application Routes

- `/` - Home page (public)
- `/login` - Login page with OAuth buttons
- `/dashboard` - Protected dashboard (requires authentication)
- `/api/auth/[...nextauth]` - NextAuth API routes

## File Structure

```
src/
├── app/
│   ├── api/auth/[...nextauth]/
│   │   └── route.ts           # NextAuth API routes
│   ├── dashboard/
│   │   └── page.tsx           # Protected dashboard page
│   ├── login/
│   │   └── page.tsx           # Login page
│   └── page.tsx               # Home page
├── auth.ts                     # NextAuth configuration
```

## Testing Authentication

1. Navigate to `http://localhost:3000`
2. Click "Get Started" to go to login page
3. Choose either Google or GitHub login
4. Complete OAuth flow
5. You'll be redirected to the dashboard

## Deployment

### Environment Variables for Production

When deploying to production (Vercel, Netlify, etc.):

1. Set all environment variables in your hosting platform
2. Update `NEXTAUTH_URL` to your production URL:
```env
NEXTAUTH_URL=https://yourdomain.com
```
3. Update OAuth callback URLs in Google and GitHub to match production URLs:
   - Google: `https://yourdomain.com/api/auth/callback/google`
   - GitHub: `https://yourdomain.com/api/auth/callback/github`

## Troubleshooting

### "Configuration error" message

- Verify all environment variables are set correctly
- Ensure AUTH_SECRET is generated and set
- Check that OAuth credentials are correct

### OAuth redirect errors

- Verify callback URLs match in OAuth provider settings
- Check NEXTAUTH_URL matches your domain
- Ensure OAuth apps are not in development/testing mode

### Session not persisting

- Check browser cookies are enabled
- Verify AUTH_SECRET is consistent across restarts
- Check for HTTPS in production (required for secure cookies)

## Security Notes

- Never commit `.env.local` to version control
- Use a strong AUTH_SECRET in production
- Regularly rotate OAuth credentials
- Enable 2FA on Google and GitHub accounts used for OAuth apps

## Next Steps

- Customize the UI in the page components
- Add user profile management
- Implement role-based access control
- Add more OAuth providers
- Set up database for persistent sessions

## Resources

- [NextAuth.js Documentation](https://next-auth.js.org/)
- [Next.js Documentation](https://nextjs.org/docs)
- [Google OAuth 2.0](https://developers.google.com/identity/protocols/oauth2)
- [GitHub OAuth Apps](https://docs.github.com/en/developers/apps/building-oauth-apps)
