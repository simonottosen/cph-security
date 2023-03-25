# ---- Base Node ----
FROM node:19.8-alpine AS base
WORKDIR /app

# ---- Dependencies ----
FROM base AS dependencies  
COPY package*.json ./
RUN npm install

# ---- Copy Files/Build ----
FROM dependencies AS build  
WORKDIR /app
COPY src /app/src
COPY public /app/public
RUN npm run build

# --- Release with Alpine ----
FROM node:19.8-alpine AS release  
WORKDIR /app
RUN npm -g install serve
COPY --from=dependencies /app/package.json ./
# Install app dependencies
RUN npm install --only=production
COPY --from=build /app/build ./build
CMD ["serve", "-s", "build"]
