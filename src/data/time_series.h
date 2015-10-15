//
// Created by Karol Dzitkowski on 11.10.15.
//

#ifndef TIME_SERIES_DATA_READER_TIME_SERIES_H
#define TIME_SERIES_DATA_READER_TIME_SERIES_H

#include <string>
#include <vector>

template<typename DataType = float, typename TimeType = unsigned long long int>
class TimeSeries
{
public:
    TimeSeries(std::string name) : _name(name) {}
    ~TimeSeries() {}
    TimeSeries(const TimeSeries& other)
            : _name(other._name), _data(other._data), _time(other._time)
    {}
    TimeSeries(TimeSeries&& other)
            : _name(std::move(other._name)), _data(std::move(other._data)), _time(std::move(other._time))
    {}

public:
    std::string GetName() const { return _name; }
    size_t GetSize() const { return _time.size(); }

    void InsertData(DataType value) { _data.push_back(value); }
    void InsertTime(TimeType value) { _time.push_back(value); }

    typename std::vector<DataType>::iterator BeginData() { return _data.begin(); }
    typename std::vector<DataType>::iterator EndData() { return _data.end(); }
    typename std::vector<TimeType>::iterator BeginTime() { return _time.begin(); }
    typename std::vector<TimeType>::iterator EndTime() { return _time.end(); }
    typename std::vector<DataType>::const_iterator BeginData() const { return _data.cbegin(); }
    typename std::vector<DataType>::const_iterator EndData() const { return _data.cend(); }
    typename std::vector<TimeType>::const_iterator BeginTime() const { return _time.cbegin(); }
    typename std::vector<TimeType>::const_iterator EndTime() const { return _time.cend(); }

public:
    bool Equal(const TimeSeries& other)
    {
        if(this->GetSize() != other.GetSize()) return false;
        auto otherTimeIt = other.BeginTime();
        auto otherDataIt = other.BeginData();
        for(auto& time : _time) if(time != *otherTimeIt++) return false;
        for(auto& data : _data) if(data != *otherDataIt++) return false;
        return true;
    }

private:
    std::string _name;
    std::vector<DataType> _data;
    std::vector<TimeType> _time;
};

#endif //TIME_SERIES_DATA_READER_TIME_SERIES_H
