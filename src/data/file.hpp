//
// Created by Karol Dzitkowski on 19.10.15.
//

#ifndef TIME_SERIES_DATA_READER_FILE_H
#define TIME_SERIES_DATA_READER_FILE_H

#include <string>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <memory>
#include <dirent.h>

#if defined(WIN32)
#  define DIR_SEPARATOR '\\'
#else
#  define DIR_SEPARATOR '/'
#endif

class File
{
public:
    File(const char* path) : _path(path), _opened(false) {}
    File(std::string path) : _path(path), _opened(false) {}
    ~File(){ Close(); }
    File(const File& other) : _path(other._path), _opened(false) {}
    File(File&& other) : _path(std::move(other._path)), _opened(false) {}

public:
    static File GetTempFile();

public:
    size_t GetSize() const;
    std::string GetPath() const;
    bool Compare(const File& other) const;
    bool Delete();
    int ReadRaw(char* rawData, size_t size);
    int WriteRaw(char* rawData, size_t size);

private:
    int Open(int flags);
    int Close();

private:
    char* GetCurrentPath() const;
    int OpenFile(const char* path, int flags) const;
    int CreateDir(char* nameOfDirectory) const;
    int CreateFile(char* nameOfFile) const;
    int CloseFile(int fileDescriptor) const;
    int SetFileSizeWithTruncate(int fileDescriptor, off_t size) const;
    int SetFileSizeWithLseek(int fileDescriptor, off_t size) const;
    int CreateFileWithSizeUsingTruncate(char* nameOfFile, off_t size) const;
    int CreateFileWithSizeUsingLseek(char* nameOfFile, off_t size) const;
    int CreateSymbolicLink(char* nameOfSymbolicLink, char* pathToFile) const;
    int ChangeDirectory(char* path) const;
    int SafeRead(int fd, char* buf, size_t size) const;
    ssize_t SafeWrite(int fd, char* buf, size_t size) const;
    ssize_t SafeWriteLine(int fd, char* buf, size_t size) const;
    char* ReadLine(int fd) const;
    FILE* OpenStream(char* fileName, char* mode) const;
    int CloseStream(FILE* file) const;
    DIR* OpenDirectory(char* path) const;
    int CloseDirectory(DIR* dirp) const;
    struct dirent* ReadDirectory(DIR *dirp) const;
    void PrintFileStats(struct dirent *dp) const;
    int LockWholeFile(int fileDesc) const;
    int UnlockWholeFile(int fileDesc) const;
    char* ReadLineFromStream(FILE* file) const;
    struct dirent* GetFileDirent(char* path, char* name) const;
    char* AllocCombinedPath(const char *path1, const char *path2) const;
    void CombinePaths(char *destination, const char *path1, const char *path2) const;
    int CreateDirInPathAndBackToCurrentDir(char* dirName, char* path, char* currentDir) const;
    int CreateDirInPath(char* dirName, char* path) const;
    int CreateDirIfNotExists(char* dirName) const;

private:
    std::string _path;
    bool _opened;
    int _flags;
    int _fd;
};


#endif //TIME_SERIES_DATA_READER_FILE_H
