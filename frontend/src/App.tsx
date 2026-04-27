import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Playground from './pages/Playground'
import Pipelines from './pages/Pipelines'
import Run from './pages/Run'
import Corpus from './pages/Corpus'
import Settings from './pages/Settings'

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen bg-gray-950 text-gray-100">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Navigate to="/playground" replace />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/pipelines" element={<Pipelines />} />
            <Route path="/run" element={<Run />} />
            <Route path="/corpus" element={<Corpus />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
