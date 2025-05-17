// src/components/SearchBar/SearchBar.tsx
import React, { useState } from 'react';
import { Search } from 'lucide-react';
import './SearchBar.css';

interface SearchBarProps {
  onSearch: (studentId: string) => void;
  placeholder?: string;
}

const SearchBar: React.FC<SearchBarProps> = ({ 
  onSearch, 
  placeholder = "输入学号搜索..." 
}) => {
  const [searchTerm, setSearchTerm] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchTerm.trim()) {
      onSearch(searchTerm.trim());
    }
  };

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder={placeholder}
        className="search-input"
      />
      <button type="submit" className="search-button">
        <Search className="search-icon" />
      </button>
    </form>
  );
};

export default SearchBar;