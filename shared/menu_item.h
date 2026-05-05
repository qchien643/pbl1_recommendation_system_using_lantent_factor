#ifndef MENU_ITEM_H
#define MENU_ITEM_H

// menu_item.h — Struct MenuItem dung chung (neu can shared shape)
// See .claude/knowledge/03-menu-codes.md

struct MenuItem {
    char  code[4];       // "P01\0"
    char  nameDisplay[50];
    float price;
    char  category;      // 'P','B','C','G','A','D','T'
};

#endif
