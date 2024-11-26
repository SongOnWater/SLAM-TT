// Unity Script to Read JSON and Interpolate Ball Position in Millimeters
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;
using System.IO;

public class BallInterpolation : MonoBehaviour
{
    public string filename = "bounce_positions";
    public GameObject ballObject;
    public GameObject bounceMarkerPrefab; // Prefab to spawn at each bounce spot
    public Transform paddleLeftTransform;
    public Transform paddleRightTranform;
    public float frameRate = 120f; // Assuming video frame rate is 30 FPS
    public bool previewBallPlacement = false;
    public bool receivingPaddleLeft = false; // Variable to determine who served first
    public bool loop = true;

    // Table dimensions in millimeters (from top-left origin)
    private const float TABLE_WIDTH_MM = 2740f;
    private const float TABLE_LENGTH_MM = 1525f;

    // Table offset to account for the table center being the origin in the Unity scene
    private const float TABLE_OFFSET_X = TABLE_WIDTH_MM / 2f / 1000f; // Convert to meters
    private const float TABLE_OFFSET_Z = TABLE_LENGTH_MM / 2f / 1000f; // Convert to meters

    private List<BounceData> bounces;
    private int currentBounceIndex = 0;
    private float timePerFrame;
    private float elapsedTime = 0f;

    [System.Serializable]
    public class BounceData
    {
        public int frame;
        public float[] position; // position[0] is x (width), position[1] is y (length)
    }

    private void Start()
    {
        timePerFrame = 1f / frameRate;
        LoadJson();
        SpawnBounceMarkers();
    }

    private void LoadJson()
    {
        TextAsset jsonFile = Resources.Load<TextAsset>(filename);
        BounceDataWrapper data = JsonConvert.DeserializeObject<BounceDataWrapper>(jsonFile.text);
        bounces = data.bounces;
    }

    private void SpawnBounceMarkers()
    {
        if (bounces == null || bounces.Count == 0)
        {
            return;
        }

        if (previewBallPlacement)
        {
            foreach (BounceData bounce in bounces)
            {
                // Convert position from mm to Unity units (meters) and adjust for table offset
                Vector3 position = new Vector3((bounce.position[0] / 1000f) - TABLE_OFFSET_X, 0.1f, (bounce.position[1] / 1000f) - TABLE_OFFSET_Z);
                Instantiate(bounceMarkerPrefab, position, Quaternion.identity);
            }
        }
    }

    private void Update()
    {
        if (bounces == null || bounces.Count == 0)
        {
            return;
        }

        if (currentBounceIndex >= bounces.Count - 1 && loop)
            currentBounceIndex = 0;

        elapsedTime += Time.deltaTime;
        float timeFraction = elapsedTime / (timePerFrame * (bounces[currentBounceIndex + 1].frame - bounces[currentBounceIndex].frame));

        if (timeFraction >= 1f)
        {
            // Move to next bounce point
            currentBounceIndex++;
            elapsedTime = 0f;
            receivingPaddleLeft = !receivingPaddleLeft;
        }
        else
        {
            // Determine target paddle based on who served first
            Transform targetPaddle = receivingPaddleLeft ? paddleLeftTransform : paddleRightTranform;

            // Interpolate between current and next bounce, adjusting for table offset
            Vector3 startPosition = new Vector3((bounces[currentBounceIndex].position[0] / 1000f) - TABLE_OFFSET_X, 0.1f, (bounces[currentBounceIndex].position[1] / 1000f) - TABLE_OFFSET_Z);
            Vector3 endPosition = new Vector3((bounces[currentBounceIndex + 1].position[0] / 1000f) - TABLE_OFFSET_X, 0.1f, (bounces[currentBounceIndex + 1].position[1] / 1000f) - TABLE_OFFSET_Z);

            float ratioToPaddle = 0.2f; // Ratio of time spent moving towards the paddle

            if (timeFraction <= ratioToPaddle)
            {
                // Move towards the paddle first
                Vector3 paddlePosition = targetPaddle.position;
                float paddleFraction = timeFraction / ratioToPaddle;
                ballObject.transform.position = Vector3.Lerp(startPosition, paddlePosition, paddleFraction);
            }
            else
            {
                // Move from paddle to the next bounce point
                float remainingFraction = (timeFraction - ratioToPaddle) / (1f - ratioToPaddle);
                Vector3 interpolatedPosition = Vector3.Lerp(targetPaddle.position, endPosition, remainingFraction);

                // Add some bounce effect in Y direction
                float maxBounceHeight = 0.3f; // Set the maximum height for the bounce (in meters)
                float midPointFraction = 0.5f;
                float bounceHeight = Mathf.Sin(Mathf.PI * remainingFraction) * maxBounceHeight;
                interpolatedPosition.y += bounceHeight * Mathf.Clamp01(1f - Mathf.Abs(remainingFraction - midPointFraction) * 2f);

                ballObject.transform.position = interpolatedPosition;
            }
        }
    }

    [System.Serializable]
    public class BounceDataWrapper
    {
        public List<BounceData> bounces;
    }
}
