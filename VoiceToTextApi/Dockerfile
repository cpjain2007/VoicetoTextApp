FROM node:20-bookworm-slim

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY server.js speaker-profiles.json ./

ENV NODE_ENV=production
EXPOSE 8080

CMD ["node", "server.js"]
