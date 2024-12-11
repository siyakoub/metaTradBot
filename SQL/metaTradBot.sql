drop database if exists metaTradBot;
create database metaTradBot;
use metaTradBot;

-- Création de la table user
drop table if exists user;
create table user(
    userID int not null auto_increment primary key,
    name char(100) not null,
    firstName char(100) not null,
    password char(100) not null,
    email char(254) not null
);

-- Création de la table telegramAPICredential
drop table if exists telegramAPICredential;
create table telegramAPICredential(
    telegramAPICredential int not null auto_increment primary key,
    botToken char(50) not null,
    telegramUserID int not null,
    userID int not null,
    -- Ajout de la contrainte UNIQUE sur userID pour garantir la relation 1:1
    constraint unique_userID unique (userID),
    -- Définition de la clé étrangère pour lier userID à la table user
    foreign key (userID) references user(userID)
);

drop table if exists metaTraderAPICredential;
create table metaTraderAPICredential(
    metaTraderAPICredentialID int not null auto_increment primary key,
    accountNumber int not null,
    password char(100) not null,
    server char(100) not null,
    userID int not null,
    foreign key (userID) references user(userID)
);