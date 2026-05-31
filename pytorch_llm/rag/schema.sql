-- ═══════════════════════════════════════════════════════════════
-- PostgreSQL + pgvector schema для Chemistry Tutor
-- ═══════════════════════════════════════════════════════════════

-- Включаем расширение pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- ═══════════════════════════════════════════════════════════════
-- Таблица: chemistry_tasks
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS chemistry_tasks (
    -- Основные поля
    id SERIAL PRIMARY KEY,
    category VARCHAR(100) NOT NULL DEFAULT 'другое',
    difficulty VARCHAR(20) NOT NULL DEFAULT 'medium',
    
    -- Содержимое задачи
    question TEXT NOT NULL,
    solution TEXT,
    answer TEXT NOT NULL,
    explanation TEXT,
    
    -- Метаданные
    keywords TEXT[],
    tags TEXT[],
    
    -- Векторное представление вопроса (384 размерность для multilingual-MiniLM-L12-v2)
    question_embedding vector(384),
    
    -- Статистика использования
    usage_count INTEGER DEFAULT 0,
    success_rate FLOAT DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════
-- Индексы для быстрого поиска
-- ═══════════════════════════════════════════════════════════════

-- Векторный индекс (IVFFlat) для косинусного сходства
-- lists = 100 хорошо работает для 10k-100k записей
CREATE INDEX IF NOT EXISTS chemistry_tasks_embedding_idx 
ON chemistry_tasks 
USING ivfflat (question_embedding vector_cosine_ops)
WITH (lists = 100);

-- Индексы для фильтрации
CREATE INDEX IF NOT EXISTS chemistry_tasks_category_idx 
ON chemistry_tasks(category);

CREATE INDEX IF NOT EXISTS chemistry_tasks_difficulty_idx 
ON chemistry_tasks(difficulty);

CREATE INDEX IF NOT EXISTS chemistry_tasks_created_at_idx 
ON chemistry_tasks(created_at DESC);

-- GIN индекс для полнотекстового поиска по массивам
CREATE INDEX IF NOT EXISTS chemistry_tasks_keywords_idx 
ON chemistry_tasks USING GIN(keywords);

CREATE INDEX IF NOT EXISTS chemistry_tasks_tags_idx 
ON chemistry_tasks USING GIN(tags);

-- ═══════════════════════════════════════════════════════════════
-- Триггер для обновления updated_at
-- ═══════════════════════════════════════════════════════════════

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_chemistry_tasks_updated_at 
BEFORE UPDATE ON chemistry_tasks
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ═══════════════════════════════════════════════════════════════
-- Функции для аналитики
-- ═══════════════════════════════════════════════════════════════

-- Получить статистику по категориям
CREATE OR REPLACE FUNCTION get_category_stats()
RETURNS TABLE(
    category VARCHAR,
    task_count BIGINT,
    avg_usage FLOAT,
    avg_success_rate FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.category,
        COUNT(*) as task_count,
        AVG(t.usage_count)::FLOAT as avg_usage,
        AVG(t.success_rate)::FLOAT as avg_success_rate
    FROM chemistry_tasks t
    GROUP BY t.category
    ORDER BY task_count DESC;
END;
$$ LANGUAGE plpgsql;

-- Поиск похожих задач с фильтрацией
CREATE OR REPLACE FUNCTION search_similar_tasks(
    query_embedding vector(384),
    search_category VARCHAR DEFAULT NULL,
    search_difficulty VARCHAR DEFAULT NULL,
    result_limit INTEGER DEFAULT 5
)
RETURNS TABLE(
    id INTEGER,
    question TEXT,
    answer TEXT,
    solution TEXT,
    category VARCHAR,
    difficulty VARCHAR,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.id,
        t.question,
        t.answer,
        t.solution,
        t.category,
        t.difficulty,
        1 - (t.question_embedding <=> query_embedding) AS similarity
    FROM chemistry_tasks t
    WHERE (search_category IS NULL OR t.category = search_category)
      AND (search_difficulty IS NULL OR t.difficulty = search_difficulty)
    ORDER BY t.question_embedding <=> query_embedding
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- ═══════════════════════════════════════════════════════════════
-- Комментарии к таблице и полям
-- ═══════════════════════════════════════════════════════════════

COMMENT ON TABLE chemistry_tasks IS 
'Таблица с химическими задачами и векторными эмбеддингами для RAG';

COMMENT ON COLUMN chemistry_tasks.question_embedding IS 
'Векторное представление вопроса (384D) для семантического поиска';

COMMENT ON COLUMN chemistry_tasks.usage_count IS 
'Количество раз, когда задача была использована в RAG';

COMMENT ON COLUMN chemistry_tasks.success_rate IS 
'Процент успешных ответов учеников на эту задачу (0.0-1.0)';

-- ═══════════════════════════════════════════════════════════════
-- Примеры запросов
-- ═══════════════════════════════════════════════════════════════

-- 1. Векторный поиск похожих задач
-- SELECT * FROM search_similar_tasks(
--     '[0.1, 0.2, ...]'::vector(384),
--     'уравнения_реакций',
--     'easy',
--     3
-- );

-- 2. Топ самых используемых задач
-- SELECT question, usage_count
-- FROM chemistry_tasks
-- ORDER BY usage_count DESC
-- LIMIT 10;

-- 3. Задачи с низким success_rate (сложные для учеников)
-- SELECT question, success_rate, category
-- FROM chemistry_tasks
-- WHERE success_rate < 0.5
-- ORDER BY success_rate ASC;

-- 4. Статистика по категориям
-- SELECT * FROM get_category_stats();

-- 5. Поиск по ключевым словам
-- SELECT question, keywords
-- FROM chemistry_tasks
-- WHERE keywords @> ARRAY['водород', 'кислород'];
