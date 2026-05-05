#ifndef CONSTANTS_H
#define CONSTANTS_H

// Kich thuoc
const int MAX_MENU    = 20;
const int MAX_ORDERS  = 1000;
const int MAX_ITEMS   = 5;
const int MAX_CLIENTS = 20;
const int MAX_USERS   = 1000;
const int MAX_TXN     = 5000;   // transactions persistent xuyen ca
const int NAME_LEN    = 40;
const int DESC_LEN    = 80;

// Latent Factor Model
const int   K        = 10;
const float LR       = 0.01f;
const float REG      = 0.02f;
const int   MAX_ITER = 50;
const float MIN_DELTA = 1e-4f;
const int   PATIENCE  = 10;

// Business
const float DISCOUNT_THRESHOLD = 2000000.0f;
const float DISCOUNT_RATE      = 0.25f;

// Network
const int DEFAULT_PORT = 8888;

#endif
