# Build stage
FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN npm ci || npm i
COPY . .
RUN npm run build

# Serve stage
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Simple default site
RUN printf "server { listen 80; root /usr/share/nginx/html; index index.html; location / { try_files \$uri /index.html; } }\n" > /etc/nginx/conf.d/default.conf
EXPOSE 80
