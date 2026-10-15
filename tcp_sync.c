#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>

int tcp_connect_sync(int rank, const char *peer_ip, int port) {
    if (rank == 0) {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        int opt = 1;
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;
        if (bind(s, (struct sockaddr*)&addr, sizeof(addr)) < 0) { close(s); return -1; }
        if (listen(s, 1) < 0) { close(s); return -2; }
        printf("[Rank 0] TCP rendezvous listening on port %d...\n", port);
        fflush(stdout);
        int client = accept(s, NULL, NULL);
        close(s);
        return client;
    } else {
        int s = socket(AF_INET, SOCK_STREAM, 0);
        struct sockaddr_in addr;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, peer_ip, &addr.sin_addr);
        printf("[Rank 1] Connecting to TCP rendezvous %s:%d...\n", peer_ip, port);
        fflush(stdout);
        while (connect(s, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            usleep(50000);
        }
        return s;
    }
}

int tcp_sync_barrier(int fd) {
    char token = 0x5A;
    if (send(fd, &token, 1, 0) != 1) return -1;
    char ack = 0;
    if (recv(fd, &ack, 1, MSG_WAITALL) != 1) return -2;
    return 0;
}

int tcp_sync_exchange(int fd, const void *send_buf, void *recv_buf, int len) {
    int sent = 0;
    while (sent < len) {
        int n = send(fd, (const char*)send_buf + sent, len - sent, 0);
        if (n <= 0) return -1;
        sent += n;
    }
    int recvd = 0;
    while (recvd < len) {
        int n = recv(fd, (char*)recv_buf + recvd, len - recvd, MSG_WAITALL);
        if (n <= 0) return -2;
        recvd += n;
    }
    return 0;
}

uint64_t current_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}
