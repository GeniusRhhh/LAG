# Qt聊天客户端完整代码 - 问题修复版

## 1. mainwindow.h (完整版本)

```cpp
#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTcpSocket>
#include <QTimer>
#include <QNetworkProxy>
#include "Protocol.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    // 按钮点击事件（需要在UI设计器中关联）
    void on_connectBtn_clicked();    // 连接/断开服务器
    void on_sendBtn_clicked();       // 发送消息
    void on_fileBtn_clicked();       // 发送文件

    // 网络相关事件（手动连接，避免命名冲突）
    void handleSocketConnected();      // 连接成功
    void handleSocketDisconnected();   // 断开连接
    void handleSocketReadyRead();      // 收到消息
    void handleSocketError(QAbstractSocket::SocketError err);  // 网络错误

private:
    Ui::MainWindow *ui;
    QTcpSocket *socket;  // TCP socket对象
    bool isConnected;    // 是否已连接
    QString nickname;    // 用户名
    
    // 辅助方法
    void updateConnectionState(bool connected);
    void displayMessage(const QString &message);
    void processMessage(MessageHeader* header, const QByteArray& content);
};

#endif // MAINWINDOW_H
```

## 2. Protocol.h (完整版本)

```cpp
#pragma once
#include <cstdint>

// 消息类型（明确指定枚举底层类型，确保跨平台大小一致）
enum class MessageType : uint16_t {
    LOGIN = 1,          // 登录
    LOGOUT = 2,         // 登出
    CHAT_MESSAGE = 3,   // 聊天消息
    FILE_REQUEST = 4,   // 文件传输请求
    FILE_DATA = 5,      // 文件数据块
    FILE_COMPLETE = 6   // 文件传输完成
};

// 消息头（每个消息都必须带这个头）
// __attribute__((packed))：取消内存对齐，确保跨平台字节序一致（GCC 支持）
struct MessageHeader {
    uint16_t type;          // 消息类型（对应 MessageType 枚举）
    uint32_t length;        // 消息内容长度（不含消息头本身）
    char sender[32];        // 发送者昵称（广播时填"系统"，用户消息填用户名）
    char timestamp[20];     // 时间戳（格式：2024-05-20 15:30:45）
} __attribute__((packed));

// 文件传输信息（配合 FILE_REQUEST/FILE_DATA 消息使用）
struct FileHeader {
    char filename[256];     // 文件名（含后缀，如 "test.txt"）
    uint64_t filesize;      // 文件总大小（字节）
    uint32_t current_chunk; // 当前传输的块编号（从 0 开始）
    uint32_t total_chunks;  // 总块数
} __attribute__((packed));
```

## 3. main.cpp (完整版本)

```cpp
#include "mainwindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.show();

    return a.exec();
}
```

## 4. mainwindow.cpp (完整版本)

```cpp
#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QDateTime>
#include <QFileDialog>
#include <QFile>
#include <QProgressDialog>
#include <QTimer>
#include <QTextCursor>
#include <QNetworkProxy>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    socket(new QTcpSocket(this)),  // 初始化socket
    isConnected(false)  // 初始未连接
{
    ui->setupUi(this);
    setWindowTitle("多人聊天客户端");  // 设置窗口标题

    // 手动关联网络信号与槽函数（避免自动连接冲突）
    connect(socket, &QTcpSocket::connected, this, &MainWindow::handleSocketConnected);
    connect(socket, &QTcpSocket::disconnected, this, &MainWindow::handleSocketDisconnected);
    connect(socket, &QTcpSocket::readyRead, this, &MainWindow::handleSocketReadyRead);
    connect(socket, QOverload<QAbstractSocket::SocketError>::of(&QAbstractSocket::error),
            this, &MainWindow::handleSocketError);

    // 初始化界面（可选：设置默认值）
    ui->portEdit->setText("8888");  // 默认端口（与服务器端口一致）
    ui->ipEdit->setText("192.168.126.128");  // 默认IP（替换为你的Linux IP）

    // 禁用代理（解决代理错误问题）
    socket->setProxy(QNetworkProxy::NoProxy);

    // 初始化连接状态
    updateConnectionState(false);
}

MainWindow::~MainWindow()
{
    if (isConnected) {
        socket->disconnectFromHost();
    }
    delete ui;
}

// 辅助方法：更新连接状态
void MainWindow::updateConnectionState(bool connected)
{
    isConnected = connected;
    ui->connectBtn->setText(connected ? "断开连接" : "连接服务器");
    ui->sendBtn->setEnabled(connected);
    ui->fileBtn->setEnabled(connected);
    ui->msgEdit->setEnabled(connected);
}

// 辅助方法：显示消息
void MainWindow::displayMessage(const QString &message)
{
    ui->chatDisplay->append(message);
    ui->chatDisplay->moveCursor(QTextCursor::End);
}

// 连接/断开服务器按钮点击事件
void MainWindow::on_connectBtn_clicked()
{
    if (isConnected) {
        // 已连接状态：断开连接
        socket->disconnectFromHost();
        displayMessage("【系统】正在断开连接...");
        return;
    }

    // 未连接状态：连接服务器
    QString ip = ui->ipEdit->text().trimmed();
    int port = ui->portEdit->text().toInt();
    nickname = ui->nameEdit->text().trimmed();

    // 检查输入合法性
    if (ip.isEmpty() || port <= 0 || port > 65535) {
        QMessageBox::warning(this, "输入错误", "请输入有效的IP地址和端口号(1-65535)");
        return;
    }
    if (nickname.isEmpty() || nickname.length() > 30) {
        QMessageBox::warning(this, "输入错误", "昵称不能为空且长度不超过30字符");
        return;
    }

    // 开始连接
    ui->connectBtn->setText("连接中...");
    ui->connectBtn->setEnabled(false);

    displayMessage(QString("【系统】正在连接服务器 %1:%2...").arg(ip).arg(port));

    // 设置连接超时（8秒）
    QTimer::singleShot(8000, this, [this]() {
        if (!isConnected && socket->state() == QAbstractSocket::ConnectingState) {
            socket->abort();
            QMessageBox::warning(this, "连接超时",
                "连接服务器超时，请检查：\n"
                "1. 服务器是否启动\n"
                "2. IP地址和端口是否正确\n"
                "3. 防火墙设置\n"
                "4. 网络连接状态");
            ui->connectBtn->setText("连接服务器");
            ui->connectBtn->setEnabled(true);
        }
    });

    socket->connectToHost(ip, port);
}

// 发送消息按钮点击事件
void MainWindow::on_sendBtn_clicked()
{
    if (!isConnected) {
        QMessageBox::warning(this, "未连接", "请先连接服务器");
        return;
    }

    QString msg = ui->msgEdit->text().trimmed();
    if (msg.isEmpty()) return;  // 空消息不发送

    // 构造消息头
    MessageHeader header;
    header.type = (uint16_t)MessageType::CHAT_MESSAGE;  // 消息类型：聊天消息
    header.length = msg.toUtf8().size();  // 消息内容长度
    strncpy(header.sender, nickname.toUtf8().data(), 31);  // 发送者昵称
    header.sender[31] = '\0';  // 确保字符串结束
    // 时间戳（格式：HH:mm:ss）
    QDateTime now = QDateTime::currentDateTime();
    strncpy(header.timestamp, now.toString("HH:mm:ss").toUtf8().data(), 19);
    header.timestamp[19] = '\0';  // 确保字符串结束

    // 拼接消息（头 + 内容）
    QByteArray data;
    data.append((char*)&header, sizeof(header));  // 添加消息头
    data.append(msg.toUtf8());  // 添加消息内容

    // 发送消息
    socket->write(data);
    ui->msgEdit->clear();  // 清空输入框
}

// 发送文件按钮点击事件
void MainWindow::on_fileBtn_clicked()
{
    if (!isConnected) {
        QMessageBox::warning(this, "未连接", "请先连接服务器");
        return;
    }

    // 选择文件
    QString filePath = QFileDialog::getOpenFileName(this, "选择文件", "", "所有文件 (*)");
    if (filePath.isEmpty()) return;

    QFile file(filePath);
    if (!file.open(QIODevice::ReadOnly)) {
        QMessageBox::critical(this, "错误", "无法打开文件：" + filePath);
        return;
    }

    QFileInfo fileInfo(filePath);
    QProgressDialog progress("正在发送文件...", "取消", 0, 100, this);
    progress.setWindowTitle("文件传输");
    progress.setWindowModality(Qt::WindowModal);
    progress.setValue(0);

    // 1. 发送文件请求（告诉服务器和其他客户端有文件要发送）
    MessageHeader header;
    header.type = (uint16_t)MessageType::FILE_REQUEST;
    QString fileInfoStr = fileInfo.fileName() + "|" + QString::number(fileInfo.size());
    header.length = fileInfoStr.toUtf8().size();
    strncpy(header.sender, nickname.toUtf8().data(), 31);
    header.sender[31] = '\0';

    QDateTime now = QDateTime::currentDateTime();
    strncpy(header.timestamp, now.toString("HH:mm:ss").toUtf8().data(), 19);
    header.timestamp[19] = '\0';

    QByteArray data;
    data.append((char*)&header, sizeof(header));
    data.append(fileInfoStr.toUtf8());
    socket->write(data);

    // 2. 分块发送文件内容（每块4KB）
    const int chunkSize = 4096;
    char buffer[chunkSize];
    qint64 bytesRead;
    uint32_t totalChunks = (fileInfo.size() + chunkSize - 1) / chunkSize;
    uint32_t currentChunk = 0;

    while ((bytesRead = file.read(buffer, chunkSize)) > 0 && !progress.wasCanceled()) {
        // 构造文件块头
        FileHeader fileHeader;
        strncpy(fileHeader.filename, fileInfo.fileName().toUtf8().data(), 255);
        fileHeader.filename[255] = '\0';
        fileHeader.filesize = fileInfo.size();
        fileHeader.current_chunk = currentChunk;
        fileHeader.total_chunks = totalChunks;

        // 构造消息
        header.type = (uint16_t)MessageType::FILE_DATA;
        header.length = sizeof(fileHeader) + bytesRead;

        data.clear();
        data.append((char*)&header, sizeof(header));
        data.append((char*)&fileHeader, sizeof(fileHeader));
        data.append(buffer, bytesRead);

        socket->write(data);
        socket->waitForBytesWritten(100);  // 等待数据发送

        // 更新进度条
        currentChunk++;
        progress.setValue((currentChunk * 100) / totalChunks);
    }

    file.close();
    progress.setValue(100);

    // 3. 发送文件完成通知
    header.type = (uint16_t)MessageType::FILE_COMPLETE;
    header.length = fileInfo.fileName().toUtf8().size();
    data.clear();
    data.append((char*)&header, sizeof(header));
    data.append(fileInfo.fileName().toUtf8());
    socket->write(data);

    displayMessage("【系统】文件《" + fileInfo.fileName() + "》发送完成");
}

// 连接成功时触发
void MainWindow::handleSocketConnected()
{
    updateConnectionState(true);
    displayMessage("【系统】成功连接到服务器！");
    ui->connectBtn->setEnabled(true);

    // 发送登录消息（告诉服务器自己的昵称）
    MessageHeader header;
    header.type = (uint16_t)MessageType::LOGIN;
    header.length = nickname.toUtf8().size();
    strncpy(header.sender, nickname.toUtf8().data(), 31);
    header.sender[31] = '\0';

    QDateTime now = QDateTime::currentDateTime();
    strncpy(header.timestamp, now.toString("yyyy-MM-dd hh:mm:ss").toUtf8().data(), 19);
    header.timestamp[19] = '\0';

    QByteArray data;
    data.append((char*)&header, sizeof(header));
    data.append(nickname.toUtf8());
    socket->write(data);
}

// 断开连接时触发
void MainWindow::handleSocketDisconnected()
{
    updateConnectionState(false);
    displayMessage("【系统】与服务器断开连接");
    ui->connectBtn->setEnabled(true);
}

// 处理单个消息
void MainWindow::processMessage(MessageHeader* header, const QByteArray& content)
{
    QString timestamp = QString::fromUtf8(header->timestamp);
    QString sender = QString::fromUtf8(header->sender);

    // 根据消息类型处理
    switch ((MessageType)header->type) {
        case MessageType::CHAT_MESSAGE:
            // 聊天消息：显示到聊天区
            displayMessage(QString("[%1] %2: %3")
                .arg(timestamp)
                .arg(sender)
                .arg(QString::fromUtf8(content)));
            break;
        case MessageType::FILE_REQUEST:
            // 文件请求：提示有文件
            displayMessage(QString("【文件通知】%1 发送了文件：%2")
                .arg(sender)
                .arg(QString::fromUtf8(content)));
            break;
        case MessageType::FILE_COMPLETE:
            // 文件传输完成
            displayMessage(QString("【文件通知】%1 的文件《%2》传输完成")
                .arg(sender)
                .arg(QString::fromUtf8(content)));
            break;
        default:
            displayMessage(QString("【系统】收到未知类型消息: %1").arg(header->type));
            break;
    }
}

// 收到消息时触发（处理粘包问题）
void MainWindow::handleSocketReadyRead()
{
    static QByteArray buffer;  // 静态缓冲区处理粘包

    buffer.append(socket->readAll());

    // 循环处理缓冲区中的完整消息
    while (buffer.size() >= sizeof(MessageHeader)) {
        MessageHeader* header = (MessageHeader*)buffer.data();
        int totalSize = sizeof(MessageHeader) + header->length;

        if (buffer.size() < totalSize) {
            break;  // 数据不完整，等待更多数据
        }

        // 提取完整消息内容
        QByteArray content = buffer.mid(sizeof(MessageHeader), header->length);

        // 处理消息
        processMessage(header, content);

        // 移除已处理的消息
        buffer.remove(0, totalSize);
    }
}

// 网络错误时触发
void MainWindow::handleSocketError(QAbstractSocket::SocketError err)
{
    QString errorTitle = "网络连接错误";
    QString errorMsg;
    QString suggestion;

    switch(err) {
        case QAbstractSocket::ConnectionRefusedError:
            errorMsg = "连接被服务器拒绝";
            suggestion = "请检查：\n• 服务器是否正在运行\n• 端口号是否正确\n• 服务器是否允许新连接";
            break;

        case QAbstractSocket::HostNotFoundError:
            errorMsg = "找不到指定的主机";
            suggestion = "请检查：\n• IP地址是否正确\n• 网络连接是否正常\n• DNS设置是否正确";
            break;

        case QAbstractSocket::NetworkError:
            errorMsg = "网络通信错误";
            suggestion = "请检查：\n• 网络连接是否稳定\n• 防火墙是否阻止连接\n• 网络代理设置";
            break;

        case QAbstractSocket::ProxyConnectionRefusedError:
        case QAbstractSocket::ProxyProtocolError:
            errorMsg = "代理服务器连接错误";
            suggestion = "请检查：\n• 代理服务器设置\n• 尝试禁用代理连接";
            break;

        case QAbstractSocket::SocketTimeoutError:
            errorMsg = "连接超时";
            suggestion = "请检查：\n• 网络连接速度\n• 服务器响应状态\n• 防火墙设置";
            break;

        default:
            errorMsg = socket->errorString();
            suggestion = "请检查网络设置和服务器状态";
    }

    displayMessage(QString("【错误】%1: %2").arg(errorTitle).arg(errorMsg));
    QMessageBox::critical(this, errorTitle, errorMsg + "\n\n" + suggestion);

    updateConnectionState(false);
    ui->connectBtn->setEnabled(true);
}
```

## 主要修改说明

### 1. 解决信号槽连接警告
- 将网络事件槽函数从 `on_socket_*` 重命名为 `handleSocket*`
- 避免Qt自动连接机制的命名冲突

### 2. 解决网络连接问题
- 添加详细的错误处理和用户提示
- 实现8秒连接超时机制
- 提供针对性的故障排除建议

### 3. 代码优化
- 添加字符串结束符确保安全
- 改进消息处理，解决TCP粘包问题
- 添加连接状态管理
- 优化用户界面反馈

### 4. 调试建议
```bash
# Windows命令行测试网络连通性
telnet 192.168.126.128 20

# 或使用PowerShell
Test-NetConnection -ComputerName 192.168.126.128 -Port 20

# Linux服务器检查端口监听
netstat -tlnp | grep :20
```

## 服务器代码诊断建议

你的服务器代码是正确的，问题可能在于Linux系统配置。建议在服务器代码中添加更多调试信息：

```cpp
// 在服务器启动成功后添加详细信息
std::cout << "[成功] 服务器启动！端口：" << port_ << "，等待客户端连接..." << std::endl;

// 添加以下调试信息
std::cout << "[调试] 服务器绑定地址：0.0.0.0:" << port_ << std::endl;
std::cout << "[调试] 服务器Socket FD：" << server_fd_ << std::endl;
std::cout << "[调试] Epoll FD：" << epoll_fd_ << std::endl;

// 获取实际绑定的地址信息
struct sockaddr_in actual_addr;
socklen_t addr_len = sizeof(actual_addr);
if (getsockname(server_fd_, (struct sockaddr*)&actual_addr, &addr_len) == 0) {
    std::cout << "[调试] 实际监听端口：" << ntohs(actual_addr.sin_port) << std::endl;
    std::cout << "[调试] 实际监听地址：" << inet_ntoa(actual_addr.sin_addr) << std::endl;
}
```

## 解决连接问题的关键修改

### 1. 端口号问题
你的服务器运行在端口8888，但客户端默认连接端口20，已修改为8888。

### 2. 代理设置问题
添加了 `socket->setProxy(QNetworkProxy::NoProxy);` 来禁用代理，解决 "The proxy type is invalid for this operation" 错误。

### 3. 额外的调试步骤

如果仍然无法连接，请按以下步骤排查：

#### 步骤1：测试网络连通性
```bash
# 在Windows命令行中测试
telnet 192.168.126.128 8888

# 或使用PowerShell
Test-NetConnection -ComputerName 192.168.126.128 -Port 8888
```

#### 步骤2：检查Linux防火墙
```bash
# CentOS/RHEL 系统
sudo firewall-cmd --list-all
sudo firewall-cmd --add-port=8888/tcp --permanent
sudo firewall-cmd --reload

# Ubuntu 系统
sudo ufw status
sudo ufw allow 8888/tcp
```

#### 步骤3：检查服务器监听状态
```bash
# 在Linux服务器上检查
netstat -tlnp | grep 8888
ss -tlnp | grep 8888
```

#### 步骤4：如果还是无法连接，在Qt代码中添加更多调试信息
在 `on_connectBtn_clicked()` 方法中添加：
```cpp
// 在连接前添加调试信息
qDebug() << "尝试连接到:" << ip << ":" << port;
qDebug() << "Socket状态:" << socket->state();
qDebug() << "代理设置:" << socket->proxy().type();
```

通过这些修改，应该能解决你遇到的连接问题。主要是端口号不匹配和代理设置导致的问题。
