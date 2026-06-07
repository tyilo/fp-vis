FROM rust:1.96-alpine AS rust-builder

WORKDIR /build
RUN apk add --no-cache curl
RUN curl https://wasm-bindgen.github.io/wasm-pack/installer/init.sh -sSf | sh
COPY fp-vis-wasm /build/fp-vis-wasm
RUN wasm-pack build fp-vis-wasm --target web

FROM node:24-alpine AS node-builder

WORKDIR /build
RUN npm install -g corepack
RUN corepack enable
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
RUN pnpm ci
COPY . /build
COPY --from=rust-builder /build/fp-vis-wasm/pkg /build/fp-vis-wasm/pkg
RUN pnpm run build

FROM nginx:alpine

COPY --from=node-builder /build/dist /usr/share/nginx/html

CMD ["nginx", "-g", "daemon off;"]
