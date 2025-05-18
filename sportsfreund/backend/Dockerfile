FROM node:latest

WORKDIR /src/api

COPY package*.json ./

RUN npm install

COPY ../.. .

EXPOSE 3000

CMD ["node", "server.js"]
