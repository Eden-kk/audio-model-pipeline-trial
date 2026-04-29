import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Sidebar from './components/Sidebar'
import Playground from './pages/Playground'
import Pipelines from './pages/Pipelines'
import Run from './pages/Run'
import Realtime from './pages/Realtime'
import Corpus from './pages/Corpus'
import Settings from './pages/Settings'

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex h-screen bg-gray-50 text-gray-900">
        <Sidebar />
        {/* min-w-0 lets <main> shrink below the intrinsic width of its
            content (default min-width: auto in a row-flex parent would
            otherwise force long unbroken transcripts to push the column
            past the viewport instead of wrapping). */}
        <main className="flex-1 min-w-0 overflow-auto">
          <Routes>
            <Route path="/" element={<Navigate to="/playground" replace />} />
            <Route path="/playground" element={<Playground />} />
            <Route path="/pipelines" element={<Pipelines />} />
            <Route path="/run" element={<Run />} />
            <Route path="/realtime" element={<Realtime />} />
            <Route path="/corpus" element={<Corpus />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
