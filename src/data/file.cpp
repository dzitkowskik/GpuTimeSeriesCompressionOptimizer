//
// Created by ghash on 19.10.15.
//

#include "data/file.hpp"

#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

std::string File::GetPath() const { return _path; }

size_t File::GetSize() const
{
    std::ifstream stream(this->GetPath(), std::ios::binary);
    size_t size = (size_t) stream.seekg(0, std::ifstream::end).tellg();
    stream.seekg(0, std::ifstream::beg);
    return size;
}

bool File::Compare(const File &other) const
{
    auto sizeA = this->GetSize();
    auto sizeB = other.GetSize();

    if(sizeA != sizeB)
        return false;

    const size_t blockSize = 4096;
    size_t remaining = sizeA;
    std::ifstream inputA(this->GetPath(), std::ios::binary);
    std::ifstream inputB(other.GetPath(), std::ios::binary);


    while(remaining)
    {
        char bufferA[blockSize], bufferB[blockSize];
        size_t size = std::min(blockSize, remaining);

        inputA.read(bufferA, size);
        inputB.read(bufferB, size);

        auto test = memcmp(bufferA, bufferB, size);
        if(test)
        {
            for(int i=0; i < size; i++)
                if(bufferA[i] != bufferB[i])
                    return false;
        }

        remaining -= size;
    }

    return true;
}

bool File::Delete()
{
    Close();
    return remove(this->GetPath().c_str()) == 0;
}

File File::GetTempFile()
{
    std::string name = "/tmp/ddj_temp_XXXXXX";
    return File(mktemp(const_cast<char*>(name.c_str())));
}

int File::ReadRaw(char* data, size_t size)
{
    Open(O_RDONLY);
    ssize_t charRead = SafeRead(_fd, data, size);
    if (charRead != size) return -1;
    return 0;
}

int File::WriteRaw(char *rawData, size_t size)
{
    Open(O_WRONLY | O_APPEND | O_CREAT);
    ssize_t charWrote = SafeWrite(_fd, rawData, size);
    if (charWrote != size) return -1;
    return 0;
}


int File::Open(int flags)
{
    if(_opened && _flags == flags) return 0;
    if(_opened) Close();
    _fd = OpenFile(_path.c_str(), flags);
    _opened = true;
    _flags = flags;
    return _fd;
}


int File::Close()
{
    if(_opened)
    {
        _opened = false;
        _flags = 0;
        return CloseFile(_fd);
    }
    return 0;
}
