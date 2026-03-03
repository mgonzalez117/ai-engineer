FROM node:20-alpine

WORKDIR /app

# Dépendances d'abord (meilleure mise en cache)
COPY frontend/package*.json ./
#RUN npm ci
RUN npm install

# Copie le reste du code front
COPY frontend/ ./

EXPOSE 3200

CMD ["npm", "run", "start", "--", "--host", "0.0.0.0", "--port", "3200"]
