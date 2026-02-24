const alveoli = [
  [90, 200], [100, 240], [80, 260], [110, 270], [120, 300], [140, 310],
  [75, 230], [130, 280], [150, 295], [160, 315],
  [310, 200], [300, 240], [320, 260], [290, 270], [280, 300], [260, 310],
  [325, 230], [270, 280], [250, 295], [240, 315],
];

const LungWireframe = () => (
  <svg
    viewBox="0 0 400 350"
    className="lung-glow w-full h-full"
    fill="none"
    stroke="hsl(24, 100%, 50%, 0.2)"
    strokeWidth="0.8"
  >
    {/* Trachea */}
    <path d="M200 30 L200 100" strokeWidth="1.2" />
    <path d="M192 30 L192 95 Q192 105 185 115" />
    <path d="M208 30 L208 95 Q208 105 215 115" />

    {/* Left bronchi */}
    <path d="M185 115 Q160 140 130 155" />
    <path d="M185 115 Q165 150 140 175" />
    <path d="M130 155 Q110 165 95 180" />
    <path d="M140 175 Q120 195 105 215" />
    <path d="M95 180 Q80 200 75 225" />
    <path d="M105 215 Q90 240 85 260" />

    {/* Right bronchi */}
    <path d="M215 115 Q240 140 270 155" />
    <path d="M215 115 Q235 150 260 175" />
    <path d="M270 155 Q290 165 305 180" />
    <path d="M260 175 Q280 195 295 215" />
    <path d="M305 180 Q320 200 325 225" />
    <path d="M295 215 Q310 240 315 260" />

    {/* Left lung outline */}
    <path
      d="M180 105 Q140 110 105 140 Q65 180 55 230 Q48 270 60 300 Q75 330 120 335 Q160 338 185 320 Q195 310 195 290"
      strokeWidth="1"
      stroke="hsl(24, 100%, 50%, 0.15)"
    />

    {/* Right lung outline */}
    <path
      d="M220 105 Q260 110 295 140 Q335 180 345 230 Q352 270 340 300 Q325 330 280 335 Q240 338 215 320 Q205 310 205 290"
      strokeWidth="1"
      stroke="hsl(24, 100%, 50%, 0.15)"
    />

    {/* Alveoli dots */}
    {alveoli.map((coords, i) => (
      <circle
        key={i}
        cx={coords[0]}
        cy={coords[1]}
        r="2"
        fill="hsl(24, 100%, 50%, 0.1)"
        stroke="hsl(24, 100%, 50%, 0.15)"
        strokeWidth="0.5"
      />
    ))}
  </svg>
);

export default LungWireframe;
