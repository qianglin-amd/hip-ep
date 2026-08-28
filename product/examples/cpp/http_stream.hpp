// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// A minimal HTTP/1.1 POST with a line callback, so the chat sample builds on Windows
// and Linux with no package manager and no dependency beyond the platform sockets.
//
// This is deliberately small rather than general. It speaks plaintext HTTP to a local
// oga-server, which is the normal deployment, and it does not implement TLS, redirects,
// proxies or chunked-encoding edge cases. In an application, replace this header with
// whatever HTTP client you already link (libcurl, cpp-httplib, WinHTTP, Qt, Boost.Beast)
// and keep chat_stream.cpp as it is: the protocol work is in the caller, not here.

#ifndef HIPEP_EXAMPLES_HTTP_STREAM_HPP
#define HIPEP_EXAMPLES_HTTP_STREAM_HPP

#include <functional>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_t = SOCKET;
#define HIPEP_INVALID_SOCKET INVALID_SOCKET
#define HIPEP_CLOSE_SOCKET closesocket
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
#define HIPEP_INVALID_SOCKET (-1)
#define HIPEP_CLOSE_SOCKET ::close
#endif

namespace hipep {

/// host:port parsed out of an "http://host:port" base URL.
struct Endpoint {
  std::string host;
  std::string port;
};

inline Endpoint parse_base_url(const std::string &base_url) {
  std::string rest = base_url;
  const size_t scheme = rest.find("://");
  if (scheme != std::string::npos) {
    if (rest.compare(0, scheme, "https") == 0) {
      throw std::runtime_error("https is not supported by this sample; use a local http:// address");
    }
    rest = rest.substr(scheme + 3);
  }
  const size_t slash = rest.find('/');
  if (slash != std::string::npos) {
    rest = rest.substr(0, slash);
  }
  const size_t colon = rest.rfind(':');
  if (colon == std::string::npos) {
    return {rest, "80"};
  }
  return {rest.substr(0, colon), rest.substr(colon + 1)};
}

class Session {
 public:
  Session() {
#ifdef _WIN32
    WSADATA data;
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
      throw std::runtime_error("WSAStartup failed");
    }
#endif
  }

  ~Session() {
#ifdef _WIN32
    WSACleanup();
#endif
  }

  Session(const Session &) = delete;
  Session &operator=(const Session &) = delete;

  /// POST `body` as JSON and invoke `on_line` for every response line.
  ///
  /// Returns the HTTP status code. `on_line` sees the response body only, split on
  /// newlines, which is exactly the shape server-sent events arrive in.
  int post_lines(const std::string &base_url, const std::string &path, const std::string &body,
                 const std::function<void(const std::string &)> &on_line) const {
    const Endpoint endpoint = parse_base_url(base_url);

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    addrinfo *resolved = nullptr;
    if (getaddrinfo(endpoint.host.c_str(), endpoint.port.c_str(), &hints, &resolved) != 0) {
      throw std::runtime_error("cannot resolve " + endpoint.host + ":" + endpoint.port);
    }

    socket_t sock = HIPEP_INVALID_SOCKET;
    for (addrinfo *candidate = resolved; candidate != nullptr; candidate = candidate->ai_next) {
      sock = ::socket(candidate->ai_family, candidate->ai_socktype, candidate->ai_protocol);
      if (sock == HIPEP_INVALID_SOCKET) {
        continue;
      }
      if (::connect(sock, candidate->ai_addr, static_cast<int>(candidate->ai_addrlen)) == 0) {
        break;
      }
      HIPEP_CLOSE_SOCKET(sock);
      sock = HIPEP_INVALID_SOCKET;
    }
    freeaddrinfo(resolved);

    if (sock == HIPEP_INVALID_SOCKET) {
      throw std::runtime_error("cannot connect to " + endpoint.host + ":" + endpoint.port +
                               " (is oga-server running?)");
    }

    const std::string request = "POST " + path + " HTTP/1.1\r\n" + "Host: " + endpoint.host + ":" +
                                endpoint.port + "\r\n" + "Content-Type: application/json\r\n" +
                                "Content-Length: " + std::to_string(body.size()) + "\r\n" +
                                "Connection: close\r\n" + "Accept: text/event-stream\r\n\r\n" + body;

    if (!send_all(sock, request)) {
      HIPEP_CLOSE_SOCKET(sock);
      throw std::runtime_error("failed to send the request");
    }

    // Headers are consumed first, then every complete line of the body is handed to the
    // callback as it arrives. Streaming only works if we never wait for the whole body.
    std::string buffer;
    bool in_body = false;
    int status = 0;
    char chunk[4096];

    for (;;) {
      const int received = ::recv(sock, chunk, static_cast<int>(sizeof(chunk)), 0);
      if (received <= 0) {
        break;
      }
      buffer.append(chunk, static_cast<size_t>(received));

      if (!in_body) {
        const size_t split = buffer.find("\r\n\r\n");
        if (split == std::string::npos) {
          continue;
        }
        const std::string headers = buffer.substr(0, split);
        if (headers.compare(0, 5, "HTTP/") == 0) {
          status = std::atoi(headers.c_str() + 9);
        }
        buffer.erase(0, split + 4);
        in_body = true;
      }

      size_t newline;
      while ((newline = buffer.find('\n')) != std::string::npos) {
        std::string line = buffer.substr(0, newline);
        buffer.erase(0, newline + 1);
        if (!line.empty() && line.back() == '\r') {
          line.pop_back();
        }
        on_line(line);
      }
    }
    if (!buffer.empty() && in_body) {
      on_line(buffer);
    }

    HIPEP_CLOSE_SOCKET(sock);
    return status;
  }

 private:
  static bool send_all(socket_t sock, const std::string &data) {
    size_t sent = 0;
    while (sent < data.size()) {
      const int written =
          ::send(sock, data.data() + sent, static_cast<int>(data.size() - sent), 0);
      if (written <= 0) {
        return false;
      }
      sent += static_cast<size_t>(written);
    }
    return true;
  }
};

}  // namespace hipep

#endif  // HIPEP_EXAMPLES_HTTP_STREAM_HPP
